### Collecting Demonstration (IMPT)
Precollected demonstrations can be downloaded at : https://drive.google.com/drive/folders/1Eyo-YH_znP2zQZ1wD6jAoyTwFLF9lqtH?usp=share_link
Download the whole folder and put in the the project's root working directory

To manually collect human demonstrations. Go to collect_human_demo.py and uncomment any of these lines depending on what you need
```
if __name__ == "__main__":

    # Collect all demos
    # collect_human_demos(num_types_demo=4,demoset_size=10)

    # Inspect a specific demo
    # demo = load_and_inspect_demo(demoset_id=0, demo_id=0)
    # print(len(demo))

    # change any sets if demos not good
    from data import GameObjective
    collect_human_demos_for(GameObjective.REACH_GOAL, 4)
    
    # Replay a demo with its configuration (for sanity check)
    # replay_demo_with_config(demoset_id=0, demo_id=0)
```

```bash
python collect_human_demo

```
### Training Instant Policy Agent

To train the **Instant Policy** agent instead of the default model, add the `--ip` flag when running the training script.  

```bash
python train.py --ip

```
### Evaluating Instant Policy Agent

To evaluate the **Instant Policy** agent instead of the default model, add the `--ip` flag when running the evaluate script.  

```bash
python eval.py --ip
```
