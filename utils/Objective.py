from utils.CustomTrainer import CustomTrainer
import optuna
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import load_metric

class bcolors:
    okblue = '\033[094'
    blue = '\033[34m'
    endc = '\033[0m'

class Objective:
  def __init__(self, 
              config, tokenizer, 
              data_collator, 
              tokenized_datasets,
              prop_clickbait):
    self.best_model = None
    self._model = None
    self.config = config
    self.tokenizer = tokenizer
    self.data_collator = data_collator
    self.tokenized_datasets = tokenized_datasets
    self.metric = load_metric("glue", "mrpc")
    self.weights = [prop_clickbait/(1+prop_clickbait),
                    1/(1+prop_clickbait)]

  def __call__(self, trial: optuna.Trial):     
  
    # Mejorable
    checkpoint = self.config['checkpoint']
    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(
                checkpoint, num_labels=2, return_dict=True)

    metric = self.metric
    def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      predictions = predictions.argmax(axis=-1)
      return metric.compute(predictions=predictions, references=labels)
    #
            
    hp = self.config['hp']
    learning_rate = trial.suggest_loguniform('learning_rate', 
                                            low=hp['learning_rate'][0], 
                                            high=hp['learning_rate'][1])
      
    epochs = trial.suggest_int('epochs', 
                              low=hp['epochs'][0], 
                              high=hp['epochs'][1])
      
    batch_size_train = trial.suggest_categorical('batch_size_train', 
                                                hp['batch_size_train'])
      
    print(bcolors.blue, 
          '\nPrueba', trial.number, 'de', self.config['n_trials'],
          '\nHiperparámetros para', self.config['checkpoint'],':', 
          '\nlearning_rate:', learning_rate,
          '\nepochs:', epochs, 
          '\nbatch_size_train:', batch_size_train, 
          bcolors.endc)

    training_args = TrainingArguments(output_dir='test',
                                      learning_rate=learning_rate,         
                                      num_train_epochs=epochs,         
                                      per_device_train_batch_size=batch_size_train, 
                                      per_device_eval_batch_size=32,
                                      save_strategy = 'no',
                                      evaluation_strategy = 'epoch',
                                      disable_tqdm=False,
                                      log_level=self.config['logger_transformers'])
                                        
    trainer = CustomTrainer(args=training_args,
                            tokenizer=self.tokenizer,
                            data_collator= self.data_collator,
                            train_dataset=self.tokenized_datasets['train'],
                            eval_dataset=self.tokenized_datasets['val'],
                            model_init=model_init,
                            compute_metrics=compute_metrics)

    trainer.set_loss_weights(self.weights)
                        
          
    result = trainer.train()

    eval_result = trainer.evaluate()     
    print(bcolors.blue, 
         '\n F1-score en test', trial.number, '=', eval_result['eval_f1'],
         '\n Accuracy en test', trial.number, '=', eval_result['eval_accuracy'], 
         '\n', bcolors.endc)

    self.trainer = trainer

    return eval_result['eval_f1']

  def callback(self, study, trial):
    if study.best_trial == trial:
        self.best_model = self.trainer.model