from utils import trainer_funcs 
import optuna
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments

class bcolors:
    okblue = '\033[094'
    blue = '\033[34m'
    endc = '\033[0m'

class Objective:
  def __init__(self, config, tokenizer, data_collator, tokenized_datasets):
    self.best_model = None
    self._model = None
    self.config = config
    self.tokenizer = tokenizer
    self.data_collator = data_collator
    self.tokenized_datasets = tokenized_datasets

  def __call__(self, trial: optuna.Trial):     
  
    # Mejorable
    checkpoint = self.config['checkpoint']
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
                checkpoint, num_labels=2, return_dict=True)
    #
            
    hp = self.config['hp']
    learning_rate = trial.suggest_loguniform('learning_rate', 
                                            low=hp['learning_rate'][0], 
                                            high=hp['learning_rate'][1])
      
    weight_decay = trial.suggest_loguniform('weight_decay', 
                                            low=hp['weight_decay'][0], 
                                            high=hp['weight_decay'][1])
      
    epochs = trial.suggest_int('epochs', 
                              low=hp['epochs'][0], 
                              high=hp['epochs'][1])
      
    batch_size_train = trial.suggest_categorical('batch_size_train', 
                                                hp['batch_size_train'])
      
    print(bcolors.blue, 
          '\nHiperparámetros test', trial.number,
          '\nlearning_rate:', learning_rate, 
          '\nweight_decay:', weight_decay, 
          '\nepochs:', epochs, 
          '\nbatch_size_train:', batch_size_train, 
          bcolors.endc)

    training_args = TrainingArguments(output_dir='test',
                                      learning_rate=learning_rate,         
                                      weight_decay=weight_decay,         
                                      num_train_epochs=epochs,         
                                      per_device_train_batch_size=batch_size_train, 
                                      per_device_eval_batch_size=32,
                                      save_strategy = 'no',
                                      disable_tqdm=False,
                                      log_level=self.config['logger_transformers'])
                                        
    trainer = Trainer(args=training_args,
                      tokenizer=self.tokenizer,
                      data_collator= self.data_collator,
                      train_dataset=self.tokenized_datasets['train'],
                      eval_dataset=self.tokenized_datasets['val'],
                      model_init=model_init,
                      compute_metrics=trainer_funcs.compute_metrics)
                        
          
    result = trainer.train()

    eval_result = trainer.evaluate()     
    print(bcolors.blue, '\n F1-score en test', trial.number,'=', eval_result['eval_f1'], '\n', bcolors.endc)
    
    self.trainer = trainer

    return eval_result['eval_f1']

  def callback(self, study, trial):
    if study.best_trial == trial:
        self.best_model = self.trainer.model