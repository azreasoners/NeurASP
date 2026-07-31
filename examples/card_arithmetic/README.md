# Card arithmetic
The Card arithmetic problem is a collection of challenging tasks that use 174x274 pixel playing card images as inputs. In each of them, the downstream result represents an arithmetic operation involving the rank and suit values of the input cards.

For each task, there are 2-, 3- and 4-card variations, and the difficulty increases accordingly. The rank values are 2-10 for the corresponding numerical ranks, and 11-14 for the Jack, Queen, King and Ace. The suit values vary depending on the task. The end result is the sum of all the card values.

## Card sum
The suits Diamonds, Clubs, Spades and Hearts are given values 0, 13, 26, 29. The card value is the _sum_ of its rank and suit value. Thus, each card is given a unique value.

## Card prodsum
The suits Diamonds, Clubs, Spades and Hearts are given values 1, 2, 3, 4. The card value is the _product_ of its rank and suit value. Thus, two different cards could be assigned the same value. E.g. the 10 of Diamonds ($10\times1=10$) has the same value as the 5 of Spades ($5\times2$). This dataset is therefore well suited for investigating reasoning shortcuts, where the neural network might learn incorrect latent labels that lead to the correct downstream label.

## How to run
To run a Card arithmetic task, you need to download the input files from https://helix.imperial.ac.uk/records/2q13m-kb120.
Store the task files in a folder called `tasks` in the `card_arithmetic` folder.
The images can be stored anywhere and are referenced by the `--image_dir` argument. 
Once you have downloaded the tasks and images, run the following command
```
python train.py
```
There are several optional arguments:
* --variant: sum_2, sum_3, sum_4, prodsum_2, prodsum_3 or prodsum_4
* --image_dir: The directory of the playing card images
* --batch_size: The batch size of the inputs
* --learning_rate: The learning rate for the neural component
* --weight_decay: Weight decay for the Adam optimizer
* --checkpoint_freq: How often the network accuracies are checked on the validation dataset
* --epochs: Number of epochs
* --seed: Seed for reproducibility
* --cpu: Whether to use the CPU (If this argument is not present, GPU is default)