import logging
import sys
import torch
import os
from tensorboardX import SummaryWriter

from sd_mhqa.hotpotqa_argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from model_envs import MODEL_CLASSES
from os.path import join
from tqdm import tqdm, trange
from sd_mhqa.hotpotqa_data_helper import DataHelper
from sd_mhqa.hotpotqa_model import UnifiedSDModel
from sd_mhqa.hotpotqa_model_utils import jd_hotpotqa_eval_model

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
# #########################################################################
# # Initialize arguments
# ##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
logger.info("Pytorch version = {}".format(torch.__version__))
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
#########################################################################
for key, value in vars(args).items():
    print('Hype-parameter\t{} = {}'.format(key, value))
#########################################################################
args = complete_default_train_parser(args)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Read Data
##########################################################################
_, _, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path, do_lower_case=True)
sep_token_id = tokenizer.sep_token_id
## ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
helper = DataHelper(sep_token_id=sep_token_id, config=args)

# Set datasets
train_dataloader = helper.hotpot_train_dataloader
dev_example_dict = helper.dev_example_dict
dev_dataloader = helper.hotpot_val_dataloader

# #########################################################################
# # Initialize Model
# ##########################################################################
model = UnifiedSDModel(config=args)
model.to(args.device)
# #########################################################################
# # Get Optimizer
# ##########################################################################
if args.max_steps > 0:
    t_total_steps = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
else:
    t_total_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
# ##########################################################################
if args.optimizer == 'RecAdam':
    optimizer, scheduler = model.rec_adam_learning_optimizer(total_steps=t_total_steps)
else:
    optimizer, scheduler = model.fixed_learning_rate_optimizers(total_steps=t_total_steps)
# ##########################################################################

if args.fp16:
    try:
        from apex import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    models, optimizer = amp.initialize([model], optimizer, opt_level=args.fp16_opt_level)
    assert len(models) == 1
    model = models[0]

# Distributed training (should be after apex fp16 initialization)
if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank,
                                                      find_unused_parameters=True)
# #########################################################################
# # launch training
# ##########################################################################
global_step = 0
loss_name = ["loss_total", "loss_span", "loss_type", "loss_sup", "loss_para"]
tr_loss, logging_loss = [0] * len(loss_name), [0]* len(loss_name)
if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(args.exp_name)

model.zero_grad()
###++++++++++++++++++++++++++++++++++++++++++
total_batch_num = len(train_dataloader)
logger.info('Total number of batches = {}'.format(total_batch_num))
eval_batch_interval_num = int(total_batch_num * args.eval_interval_ratio) + 1
logger.info('Evaluate the model by = {} batches'.format(eval_batch_interval_num ))
# #########################################################################
# # Show model information
# #########################################################################
logging.info('Model Parameter Configuration:')
for name, param in model.named_parameters():
    logging.info('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
logging.info('*' * 75)
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
start_epoch = 0
best_joint_f1 = 0.0
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
train_iterator = trange(start_epoch, start_epoch+int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
for epoch in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
        model.train()
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        loss_list = model(batch, return_yp=True)
        del batch
        if args.n_gpu > 1:
            for loss in loss_list:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if args.gradient_accumulation_steps > 1:
            for loss in loss_list:
                loss = loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss_list[0], optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            loss_list[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        for idx in range(len(loss_name)):
            if not isinstance(loss_list[idx], int):
                tr_loss[idx] += loss_list[idx].data.item()
            else:
                tr_loss[idx] += loss_list[idx]

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1
            if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                avg_loss = [(_tr_loss - _logging_loss) / (args.logging_steps*args.gradient_accumulation_steps)
                             for (_tr_loss, _logging_loss) in zip(tr_loss, logging_loss)]
                loss_str = "step[{0:6}] " + " ".join(['%s[{%d:.5f}]' % (loss_name[i], i+1) for i in range(len(avg_loss))])
                logger.info(loss_str.format(global_step, *avg_loss))
                # tensorboard
                tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                for i in range(len(loss_name)):
                    tb_writer.add_scalar(loss_name[i], (tr_loss[i] - logging_loss[i]) / (
                                args.logging_steps * args.gradient_accumulation_steps), global_step)
                logging_loss = tr_loss.copy()
        if args.max_steps > 0 and global_step > args.max_steps:
            epoch_iterator.close()
            break
        ##++++++
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ########################+++++++
        if (step + 1) % eval_batch_interval_num == 0:
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                output_pred_file = os.path.join(args.exp_name, f'pred.epoch_{epoch + 1}.step_{step + 1}.json')
                output_eval_file = os.path.join(args.exp_name, f'eval.epoch_{epoch + 1}.step_{step + 1}.txt')
                metrics, threshold = jd_hotpotqa_eval_model(args, model,
                                                        dev_dataloader, dev_example_dict,
                                                        output_pred_file, output_eval_file, args.dev_gold_file)
                if metrics['joint_f1'] >= best_joint_f1:
                    best_joint_f1 = metrics['joint_f1']
                    # torch.save({'epoch': epoch + 1,
                    #             'lr': scheduler.get_lr()[0],
                    #             'encoder': 'encoder.pkl',
                    #             'model': 'model.pkl',
                    #             'best_joint_f1': best_joint_f1,
                    #             'threshold': threshold},
                    #            join(args.exp_name, f'cached_config.bin')
                    #            )
                    logger.info(
                        'Current best joint_f1 = {} with best threshold = {}'.format(best_joint_f1, threshold))
                    for key, val in metrics.items():
                        logger.info("Current {} = {}".format(key, val))
                    logger.info('*' * 100)
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        torch.save({k: v.cpu() for k, v in model.module.encoder.state_dict().items()},
                                   join(args.exp_name, f'encoder_{epoch + 1}.step_{step + 1}.pkl'))
                        torch.save({k: v.cpu() for k, v in model.module.model.state_dict().items()},
                                   join(args.exp_name, f'model_{epoch + 1}.step_{step + 1}.pkl'))
                    else:
                        torch.save({k: v.cpu() for k, v in model.encoder.state_dict().items()},
                                   join(args.exp_name, f'encoder_{epoch + 1}.step_{step + 1}.pkl'))
                        torch.save({k: v.cpu() for k, v in model.model.state_dict().items()},
                                   join(args.exp_name, f'model_{epoch + 1}.step_{step + 1}.pkl'))

                for key, val in metrics.items():
                    tb_writer.add_scalar(key, val, epoch)
        ########################+++++++
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        output_pred_file = os.path.join(args.exp_name, f'pred.epoch_{epoch + 1}.json')
        output_eval_file = os.path.join(args.exp_name, f'eval.epoch_{epoch + 1}.txt')
        metrics, threshold = jd_hotpotqa_eval_model(args, model,
                                                dev_dataloader, dev_example_dict,
                                                output_pred_file, output_eval_file, args.dev_gold_file)
        if metrics['joint_f1'] >= best_joint_f1:
            best_joint_f1 = metrics['joint_f1']
            # torch.save({'epoch': epoch + 1,
            #             'lr': scheduler.get_lr()[0],
            #             'encoder': 'encoder.pkl',
            #             'model': 'model.pkl',
            #             'best_joint_f1': best_joint_f1,
            #             'threshold': threshold},
            #            join(args.exp_name, f'cached_config.bin')
            #            )
            logger.info('Current best joint_f1 = {} with best threshold = {}'.format(best_joint_f1, threshold))
            for key, val in metrics.items():
                logger.info("Current {} = {}".format(key, val))
            logger.info('*' * 100)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                torch.save({k: v.cpu() for k, v in model.module.encoder.state_dict().items()},
                           join(args.exp_name, f'encoder_{epoch + 1}.pkl'))
                torch.save({k: v.cpu() for k, v in model.module.model.state_dict().items()},
                           join(args.exp_name, f'model_{epoch + 1}.pkl'))
            else:
                torch.save({k: v.cpu() for k, v in model.encoder.state_dict().items()},
                           join(args.exp_name, f'encoder_{epoch + 1}.pkl'))
                torch.save({k: v.cpu() for k, v in model.model.state_dict().items()},
                           join(args.exp_name, f'model_{epoch + 1}.pkl'))

        for key, val in metrics.items():
            tb_writer.add_scalar(key, val, epoch)

if args.local_rank in [-1, 0]:
    tb_writer.close()