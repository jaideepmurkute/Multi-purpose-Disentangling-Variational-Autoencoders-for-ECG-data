import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from gvae_ecg.trainingVAE import train, test, train_unsupervised_classifier
import sys
# from utils import get_data_loader, get_model_reference, plot_training_history, min_max_scaling
from gvae_ecg.utils import *
from gvae_ecg.metrics_ecg import mutual_info_metric_shapes
import shutil

seed = 41
np.random.seed(seed)
torch.manual_seed(seed)
torch.manual_seed(seed)
if not torch.cuda.is_available():
    print("No GPU w/ CUDA visible ...")
else:
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    print("PyTorch is using GPU: ", torch.cuda.get_device_name(0))

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
use_gpu_id = 0  # Set which GPU ID to use
if cuda_available:
    if torch.cuda.device_count == 1:
        use_gpu_id = 0
    torch.cuda.set_device(use_gpu_id)


def arg_parser():
    parser = argparse.ArgumentParser(description='VAE configurations')

    parser.add_argument('--model_name', type=str, nargs='?', default='vae_beta_clinical_iid',
                        help='Unique name for the model')

    parser.add_argument('--sample_treatment', type=str, nargs='?', default='iid',
                        help='Choose from iid or group')
    parser.add_argument('--learning_type', type=str, nargs='?', default='unsupervised',
                        help='supervised or unsupervised')
    parser.add_argument('--finetune_encoder', type=bool, nargs='?', default=True,
                        help='Whether or not to update encoder weights when training classifier for unsupervised models')
    parser.add_argument('--architecture_type', type=str, nargs='?', default='cnn',
                        help='Choose from cnn or mlp')
    parser.add_argument('--model_type', type=str, nargs='?', default='other',
                        help='Choose from ibp OR other(for cnn-vae and ml-vae) OR baseline')
    parser.add_argument('--dataset_type', type=str, nargs='?', default='clinical',
                        help='simulated or clinical')
    parser.add_argument('--penalty_type', type=str, nargs='?', default='beta',
                        help='vanilla or beta')
    parser.add_argument('--beta', type=int, default=5, help='beta penalty for VAE training (default: 5)')
    parser.add_argument('--validate_during_training', type=bool, default=False,
                        help='To get test/validation set performance after each epoch.')
    parser.add_argument('--test_after_training', type=bool, default=False, help='To get test set performance at the end.')
    parser.add_argument('--log_history_w_tensorboard', type=bool, default=False,
                        help='Log progress with Tensor-board. Currently not supported')
    parser.add_argument('--train_classifier', type=bool, default=True,
                        help='Just train classifier from existing or new model(new model case is same as baseline)')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of heart segments to be classified')

    parser.add_argument('--use_lr_schedule', type=bool, nargs='?', default=True,
                        help='Whether To use learning rate decay based on ReduceLROnPlateau')
    parser.add_argument('--best_epoch_tolerance', type=int, default=3,
                        help='Epochs to wait with no improvement, for Early Stopping event')
    parser.add_argument('--epochs', type=int, default=50, help='Max. number of epochs to train VAE')
    parser.add_argument('--lr', type=float, default=3e-4, help='VAE learning rate')
    parser.add_argument('--unsup_clf_epochs', type=int, default=35,
                        help='Max. number of epochs to train classifier after unsupervised training')
    parser.add_argument('--unsup_clf_lr', type=float, default=1e-3,
                        help='learning rate for classification task training after unsupervised VAE training')
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 512)')
    parser.add_argument('--bottleneck_size', type=int, default=100, help='VAE bottleneck size (default: 100)')

    parser.add_argument('--optimizer_type', type=str, default='adam', help='Choose optimizer: adam OR SGD')
    parser.add_argument('--load_optim_from_checkpoint', type=bool, default=False,
                        help='Load optimizer - for training continuation')

    parser.add_argument('--l2_penalty', type=float, default=0.0, help='L2-weight decay penalty')

    parser.add_argument('--SGD_nesterov', type=bool, default=True, help='enables SGD Nesterov momentum')
    parser.add_argument('--SGD_momentum', type=float, default=0.9, help='SGD Nesterov momentum value')

    parser.add_argument('--samples_to_read', type=int, default=10,
                        help='How many samples to read. -1 for all samples. Useful for testing changes.')
    parser.add_argument('--print_log_every', type=int, default=20,
                        help='print loss and accuracy values after these many batches during training.')

    parser.add_argument('--device', type=object, default=device,
                        help='Device being used by PyTorch.')
    
    parser.add_argument('--alpha0', type=float, default=10,
                        help='alpha value for Beta distribution.')
    parser.add_argument('--temp', type=float, default=1.,
                        help='temp')
    parser.add_argument('--temp_prior', type=float, default=1.,
                        help='temp_prior')
    
    return parser.parse_args()

global args
args = arg_parser()

# ---------------------------------------------------------------------------------------------------------------------


def train_model(args, data_dir, plots_store_dir, model_store_dir):
    model = get_model_reference(args, model_store_dir, training_flag=True, load_from_checkpoint=False)
    optimizer, scheduler = get_optimizer(args, model_store_dir, model)

    train_scores = np.zeros(args.epochs)
    train_acc = np.zeros(args.epochs)
    NLL_log_train = []
    KLD_log_train = []
    ELBO_log_train = []
    XENTROPY_log_train = []
    ACCURACY_log_train = []

    NLL_log_train_epoch = []
    KLD_log_train_epoch = []
    ELBO_log_train_epoch = []
    XENTROPY_log_train_epoch = []
    ACCURACY_log_train_epoch = []

    NLL_log_val_epoch = []
    KLD_log_val_epoch = []
    ELBO_log_val_epoch = []
    XENTROPY_log_val_epoch = []
    ACCURACY_log_val_epoch = []

    NLL_log_val = []
    KLD_log_val = []
    ELBO_log_val = []
    XENTROPY_log_val = []
    ACCURACY_log_val = []
    all_elbo_plot_log = []
    learning_rate_log = []

    validation_scores = np.zeros(args.epochs)
    validation_acc = np.zeros(args.epochs)

    epoch_times = np.zeros(args.epochs)

    if args.learning_type == 'supervised' or args.learning_type == 'baseline':
        best_valid = -1
        # best_valid = sys.maxsize
    elif args.learning_type == 'unsupervised':
        best_valid = sys.maxsize

    train_loader = get_data_loader(args, mode='train', data_dir=data_dir)
    if args.validate_during_training:
        val_loader = get_data_loader(args, mode='validation', data_dir=data_dir)

    start = time.time()
    best_epoch_number = -1
    print("\n + Starting training ...")
    if args.learning_type == 'unsupervised':
        best_model = None
    for epoch in range(1, args.epochs + 1):
        # learning_rate_log.append(scheduler.get_lr())
        train_scores[epoch - 1], all_accuracy_train, all_NLL_train, all_KLD_train, all_elbo_train, \
            all_xentropy_train, all_elbo_plot, optimizer, scheduler = train(args, train_loader, model, optimizer,
                                                      epoch, scheduler)

        NLL_log_train_epoch.append(np.mean(all_NLL_train))
        KLD_log_train_epoch.append(np.mean(all_KLD_train))
        ELBO_log_train_epoch.append(np.mean(all_elbo_train))

        XENTROPY_log_train_epoch.append(np.mean(all_xentropy_train))
        ACCURACY_log_train_epoch.append(np.mean(all_accuracy_train))

        print("==========>Summary(Avg.) of epoch: {}".format(epoch))
        print("---->On train set: ")
        if args.model_type == 'baseline':
            print("XEntropy: {:.5f} \t Accuracy: {:.5f}".format(XENTROPY_log_train_epoch[-1],
                ACCURACY_log_train_epoch[-1]))
        else:
            print("ELBO: {:.5f} \t KLD: {:.5f} \t NLL: {:.5f} \t XEntropy: {:.5f} \t Accuracy: {:.5f}".format(
                ELBO_log_train_epoch[-1], KLD_log_train_epoch[-1], NLL_log_train_epoch[-1], XENTROPY_log_train_epoch[-1],
                ACCURACY_log_train_epoch[-1]))

        NLL_log_train += all_NLL_train
        KLD_log_train += all_KLD_train
        ELBO_log_train += all_elbo_train
        all_elbo_plot_log += all_elbo_plot
        if args.learning_type == 'supervised':
            train_acc[epoch - 1] = np.mean(all_accuracy_train)
            XENTROPY_log_train += all_xentropy_train
            ACCURACY_log_train += all_accuracy_train

        if args.validate_during_training:
            validation_scores[epoch - 1], all_accuracy_val, all_NLL_val, all_KLD_val, all_elbo_val, \
                all_xentropy_val = test(args, model_store_dir, val_loader, model, epoch,
                                        store_outputs=False)

            NLL_log_val += all_NLL_val
            KLD_log_val += all_KLD_val
            ELBO_log_val += all_elbo_val
            if args.learning_type == 'supervised':
                validation_acc[epoch - 1] = np.mean(all_accuracy_val)
                XENTROPY_log_val += all_xentropy_val
                ACCURACY_log_val += all_accuracy_val

            NLL_log_val_epoch.append(np.mean(all_NLL_val))
            KLD_log_val_epoch.append(np.mean(all_KLD_val))
            ELBO_log_val_epoch.append(np.mean(all_elbo_val))
            XENTROPY_log_val_epoch.append(np.mean(all_xentropy_val))
            ACCURACY_log_val_epoch.append(np.mean(all_accuracy_val))

        epoch_times[epoch - 1] = time.time() - start

        if args.validate_during_training:
            print("On validation set: ")
            print("ELBO: {:.5f} \t KLD: {:.5f} \t NLL: {:.5f} \t XEntropy: {:.5f} \t Accuracy: {:.5f}".format(
                ELBO_log_val_epoch[-1], KLD_log_val_epoch[-1], NLL_log_val_epoch[-1], XENTROPY_log_val_epoch[-1],
                ACCURACY_log_val_epoch[-1]))

        is_best = False
        if args.validate_during_training:
            if args.learning_type == 'supervised':
                print("validation_acc[epoch - 1]: ", validation_acc[epoch - 1])
                is_best = validation_acc[epoch - 1] > best_valid
                best_valid = max(best_valid, validation_acc[epoch - 1])
            elif args.learning_type == 'unsupervised':
                is_best = ELBO_log_val_epoch[-1] < best_valid
                if is_best:
                    best_valid = ELBO_log_val_epoch[-1]
        else:
            if args.learning_type == 'supervised':
                is_best = train_acc[epoch - 1] > best_valid
                if is_best:
                    best_valid = train_acc[epoch - 1]
                # is_best = np.mean(all_elbo_train) < best_valid
                # best_valid = np.mean(all_elbo_train)
            if args.learning_type == 'unsupervised:':
                is_best = np.mean(all_elbo_train) < best_valid
                if is_best:
                    best_valid = np.mean(all_elbo_train)
                    best_model = model

        if is_best:
            best_epoch_number = epoch
            torch.save(model.state_dict(), model_store_dir + '/' + args.model_name + '_best.pt')
            print("New best epoch found: ", epoch)

        # torch.save(model.state_dict(), model_store_dir + '/' + args.model_name + '_last_epoch.pt')
        torch.save(model.state_dict(), model_store_dir + '/' + args.model_name + '.pt')

        if (epoch - best_epoch_number) == args.best_epoch_tolerance:
            print("Best epoch not found since last {} epochs. Stopping training.".format(args.best_epoch_tolerance))
            # break


        print("Last best Epoch: ", best_epoch_number)
        print("-" * 50)
    print("model stored at: ", model_store_dir + '/' + args.model_name + '.pt')

    print("-" * 40)
    print("Best epoch: ", best_epoch_number)
    print("Average training loss: ", np.mean(train_scores))
    print("Average training accuracy: ", np.mean(train_acc))
    if args.validate_during_training:
        print("Average validation loss: ", np.mean(validation_scores))
        print("Average validation accuracy: ", np.mean(validation_acc))
    print("-" * 40)

    if args.learning_type == 'unsupervised':
        print("Training for classification task ...")
        model.train()
        classifier_lr = args.unsup_clf_lr
        classifier_max_epochs = args.unsup_clf_epochs
        optimizer = optim.Adam(model.parameters(), lr=classifier_lr, weight_decay=0.0005)
        scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
        # scheduler_interval = args.unsup_clf_scheduler_interval

        if args.finetune_encoder:
            print("Keeping whole network un-frozen. Back-propagating only through the classification branch.")
        else:
            for name, child in model.named_children():
                if 'conv' in name or 'bn' in name:
                    print("Freezing layer: {}".format(name))
                    for param in child.parameters():
                        param.requires_grad = False
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=classifier_lr)
            print("optimizer updated to only compute gradients of the un-frozen layers ...")

        xentropy_clf_log_train = []
        accuracy_clf_log_train = []
        xentropy_clf_log_val = []
        accuracy_clf_log_val = []

        best_accuracy = -1
        best_epoch = 1
        for epoch in range(1, classifier_max_epochs + 1):
            print("Starting epoch {}".format(epoch))
            model, xentropy_clf, accuracy_clf, optimizer = train_unsupervised_classifier(train_loader, model, args.dataset_type,
                                                                              optimizer, args)

            if args.validate_during_training:
                # args, model_store_dir, test_loader, model, log_likelihood, epoch, store_outputs=False
                _, all_accuracy_val, _, _, _, all_xentropy_val = test(args, model_store_dir, val_loader, model, epoch)
                model.train()
                xentropy_clf_log_val += all_xentropy_val
                accuracy_clf_log_val += all_accuracy_val

            xentropy_clf_log_train += xentropy_clf
            accuracy_clf_log_train += accuracy_clf

            epoch_accuracy = np.mean(accuracy_clf)
            if epoch_accuracy > best_accuracy:
                best_epoch = epoch
                best_accuracy = epoch_accuracy
                print("New best classification epoch found: {}. Best Accuracy: {:.5f}".format(epoch, best_accuracy))
                torch.save(model.state_dict(), model_store_dir + '/' + args.model_name + '_classification_best.pt')
            torch.save(model.state_dict(), model_store_dir + '/' + args.model_name + '_classification.pt')
            if (epoch - best_epoch) == args.best_epoch_tolerance:
                print("No best epoch found since {} epochs. Existing classifier training ...".
                      format(args.best_epoch_tolerance))
                break

            print()
        if args.validate_during_training:
            num_plots = 2
        else:
            num_plots = 1
        plt.figure()
        plt.subplot(num_plots, 1, 1)
        plt.plot(accuracy_clf_log_train, color='blue')
        plt.plot(accuracy_clf_log_val, color='green')
        plt.legend(['Train', 'Validation'])
        plt.ylabel("Classification Accuracy")
        if args.validate_during_training:
            plt.subplot(num_plots, 1, 2)
            plt.plot(xentropy_clf_log_train, color='blue')
            plt.plot(xentropy_clf_log_val, color='green')
            plt.legend(['Train', 'Validation'])
            plt.ylabel("Classification X-entropy")
        plt.savefig(fname=plots_store_dir+'/unsup_clf_training_log', dpi=400)


    all_elbo_plot_log = min_max_scaling(all_elbo_plot_log)
    plt.figure()
    plt.plot(all_elbo_plot_log)
    plt.savefig('all_elbo_plot_log')

    for normalize_plot in [True, False]:
        plot_training_history(train_scores, validation_scores, plots_store_dir, learning_rate_log,
                              NLL_log_train, NLL_log_val, KLD_log_train, KLD_log_val, ELBO_log_train,
                              ELBO_log_val, XENTROPY_log_train, XENTROPY_log_val, ACCURACY_log_train,
                              ACCURACY_log_val, args.validate_during_training, args.learning_type, NLL_log_train_epoch,
                              KLD_log_train_epoch, ELBO_log_train_epoch, XENTROPY_log_train_epoch,
                              ACCURACY_log_train_epoch, NLL_log_val_epoch, KLD_log_val_epoch, ELBO_log_val_epoch,
                              XENTROPY_log_val_epoch, ACCURACY_log_val_epoch, normalize_plot)


def test_model(args, model_store_dir, data_dir, dataset_split=''):
    model = get_model_reference(args, model_store_dir, training_flag=False, load_from_checkpoint=True)
    model.eval()
    test_loader = get_data_loader(args, mode=dataset_split, data_dir=data_dir)



    print("Evaluating on {} set: ".format(dataset_split))
    # args, val_loader, model, log_liklihood_VT, epoch
    total_avg_loss, all_accuracy, all_NLL, all_KLD, all_elbo, all_xentropy = test(args, model_store_dir, test_loader, model,
                                                                                  0, store_outputs=True)

    print("Avg. NLL: ", np.mean(all_NLL))
    print("Avg. KLD: ", np.mean(all_KLD))
    print("Avg. ELBO: ", np.mean(all_elbo))
    print("Avg. cross-entropy: ", np.mean(all_xentropy))
    print("Avg. Accuracy on {} set: {:5f}".format(args.dataset_type, np.mean(all_accuracy)))


if __name__ == '__main__':
    # -------------------------------- House-keeping ---------------------------------------------------------------
    if args.penalty_type == 'vanilla':
        args.beta = 1

    if args.dataset_type == 'simulated':
        data_store_dir = './data/simulated_dataset'
    elif args.dataset_type == 'clinical':
        data_store_dir = './data/clinical_dataset'

    model_home_dir = args.learning_type + '_' + args.penalty_type + '_' + args.architecture_type
    model_store_home = './model_store'
    model_store_dir = model_store_home + '/' + args.model_name
    # model_store_dir = model_store_home + '/' + args.learning_type+'_'+args.penalty_type+'_'+args.architecture_type.lower()
    # plots_store_home = './plots_store'
    # plots_store_dir = plots_store_home + '/' + args.model_name
    plots_store_dir = model_store_dir + '/plots'
    
    choice = int(input("Enter choice: 1] Train \t 2] Test \n"))

    if not os.path.exists(model_store_home):
        os.mkdir(model_store_home)
    if not os.path.exists(model_store_dir):
        os.mkdir(model_store_dir)
    elif os.path.exists(model_store_dir):
        model_store_dir_files = os.listdir(model_store_dir)
        if args.model_name in model_store_dir_files and choice == 1:
            new_model_name = args.model_name+'_'+str(np.random.randint(1, 100, 1)[0])
            print("Model named {} already exists. Naming new model to be trained as {}.".format(args.model_name,
                                                                                                new_model_name))
            model_name = new_model_name
    #if not os.path.exists(plots_store_home):
    #    os.mkdir(plots_store_home)
    if not os.path.exists(plots_store_dir):
        os.mkdir(plots_store_dir)

    # ------------------------------------------------------------------------------------------------------------

    if choice == 1:
        files_to_backup = ['./main.py', './trainingVAE.py', './cnn_model.py', './cnn_model_new1.py', './commonModels.py', './metrics_ecg.py', './trainingVAE_clinical.py', './utils.py']
        for file_path in files_to_backup:
            try: 
                if os.path.isfile(file_path) or os.path.isdir(file_path):
                    file_name = file_path.split('/')[-1]
                    backup_path = model_store_dir+'/'+file_name
                    print("Copying {}   to   {}".format(file_path, backup_path))
                    shutil.copy2(file_path, backup_path)
                else:
                    print("could not backup file {}, File not found at mentioned path ...".format(file_path))
            except Exception as e:
                print("Exception occurred during creating backup code files... ")
                print(e)
        
        train_model(args, data_store_dir, plots_store_dir, model_store_dir)
    elif choice == 2:
        test_model(args, model_store_dir, data_store_dir, dataset_split='test')
        plot_roc_multiclass(args, model_store_dir)

        # test_loader = get_data_loader(args, mode='test', data_dir=data_store_dir)
        # metric, marginal_entropies, cond_entropies = mutual_info_metric_shapes(model, test_loader, args.batch_size)
        # print("metric: ", metric)
        # print("marginal_entropies: ", marginal_entropies)
        # print("cond_entropies: ", cond_entropies)
    else:
        print("Invalid choice choice ...")
