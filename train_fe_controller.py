from fe_controller import FeatureExtractorController
from torch.utils.tensorboard import SummaryWriter
from models import Autoencoder
import numpy as np
import torch
import copy

# torch seed
random_seed = 42
# number of iterations before training controller
warmup_iterations = 0                       
# number of child models evaluated before each controller update
num_rollouts_per_iteration = 1
# number of PG updates before saving controller policies
save_policy_frequency = 5
# save as a string so logger can log (yeah it's hacky I know)
reward_map_fn_str = 'lambda x: np.power(10, -x)'
# compute model quality as `reward_map_fn`(validation accuracy)
reward_map_fn = eval(reward_map_fn_str)

torch.manual_seed(random_seed)
np.random.seed(random_seed)
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

dimension = 3
reps = 3
encoder_layers = [(1, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
# [(output_features, input_features), ...]
decoder_layers = [(32, 128), (16, 64), (8, 32), (4, 16), (1, 8), (1, 2, False)]
model = Autoencoder(dimension, reps, encoder_layers, decoder_layers, unet=True, use_sparsify=False)
evaluator = model.encoder
evaluator.freeze()

evaluator.to(device)
controller = FeatureExtractorController(
    encoder=evaluator,
    encoder_features=encoder_layers,
    N=4, 
    latent_space_size=torch.LongTensor([4,4,4]),
    latent_space_channels=64, 
    archspace_epochs=10, 
    reward_map_fn=reward_map_fn, 
    device=device)

logger = SummaryWriter()
logger.add_scalar('Random seed', random_seed)
logger.add_scalar('Warmup iterations', warmup_iterations)
logger.add_scalar('N rollout per iteration', num_rollouts_per_iteration)
logger.add_scalar('N training epochs', controller.archspace.epochs)
logger.add_text('Reward Mapping Function', reward_map_fn_str)

iteration = 0
print("Training controller...")
while not controller.has_converged():
    print(f"Iteration {iteration}\n\n", end="\r")

    rollouts = []
    for t in range(num_rollouts_per_iteration):
        print(f"\n\tTimestep {t}")
        print("\tLoading child...")

        model_sample, hp_sample = controller.sample()
        hp_state = controller.hpspace[tuple([int(h) for h in hp_sample])]
        hyperparameters = {'optimizer': hp_state[0], 'learning_rate': hp_state[1]}
        model_sample_ = [int(s) for s in model_sample]
        model = controller.archspace[model_sample_]

        print("\tTraining child...")
        controller.archspace.train_child(model, hyperparameters, indentation_level=2)
        controller.archspace.save_child(model)

        print("\tEvaluating child quality...")
        model_copy = copy.deepcopy(model)
        model_sample_copy, hp_sample_copy = copy.deepcopy(model_sample), copy.deepcopy(hp_sample)
        hp_state = controller.hpspace[tuple([int(h) for h in hp_sample_copy])]
        hyperparameters_copy = {'optimizer': hp_state[0], 'learning_rate': hp_state[1]}
        quality = controller.archspace.get_reward_signal(model_copy, hyperparameters_copy)
        rollouts.append([model_sample_copy, hp_sample_copy, quality])
        print(f"\tChild quality is {quality}")

    if iteration >= warmup_iterations:
        print("\tUpdating controller...")
        controller.update(rollouts)

        print("\n\n\tLoading argmax child...")
        model_sample, hp_sample = controller.argmax()
        hp_state = controller.hpspace[tuple([int(h) for h in hp_sample])]
        hyperparameters = {'optimizer': hp_state[0], 'learning_rate': hp_state[1]}
        model_sample_ = [int(s) for s in model_sample]
        model = controller.archspace[model_sample_]

        print("\tTraining argmax child...")
        controller.archspace.train_child(model, hyperparameters, indentation_level=2)

        print("\tEvaluating argmax child quality", end='\r')
        quality = controller.archspace.get_reward_signal(model, hyperparameters)
        print(f"\tArgmax child quality is {quality}")

        logger.add_scalar("Loss/argmax", quality, iteration)

        if quality < 1e-7:
            controller.converged = True

    else:
        print("\tNot updating controller!")

    average_quality = np.mean([r[2] for r in rollouts])
    logger.add_scalar("Loss/val", average_quality, iteration)
    print(f"\tAverage child quality over rollout is {average_quality}")

    # save histograms of controller policies
    for p in controller.policies['archspace']:
        params = controller.policies['archspace'][p].state_dict()['params']
        params /= torch.sum(params)
        logger.add_scalars(
            f'Parameters/Policy {p} Normalized Parameters',
            {'param %d' % i: params[i] for i in range(len(params))},
            iteration
        )
        logger.add_histogram(f'Policy {p} Normalized Parameters', params, iteration)

    # periodically save controller policy weights
    if iteration % save_policy_frequency == 0:
        print("\tSaving controller policies...")
        controller.save_policies()

    iteration += 1
    
# save final controller policy weights after convergence
controller.save_policies('fecontroller_weights_converged')