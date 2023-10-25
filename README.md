# R2H: Building Multimodal Navigation Helpers that Respond to Help Requests
**Authors:** Yue Fan, Jing Gu, Kaizhi Zheng, Xin Eric Wang (UC Santa Cruz)

**Abstract:**
Intelligent navigation-helper agents are critical as they can navigate users in unknown areas through environmental awareness and conversational ability, serving as potential accessibility tools for individuals with disabilities. In this work, we first introduce a novel benchmark, Respond to Help Requests (R2H), to promote the development of multi-modal navigation helpers capable of responding to requests for help, utilizing existing dialog-based embodied datasets. R2H mainly includes two tasks: (1) Respond to Dialog History (RDH), which assesses the helper agent's ability to generate informative responses based on a given dialog history, and (2) Respond during Interaction (RdI), which evaluates the effectiveness and efficiency of the response during consistent cooperation with a task performer. Furthermore, we explore two approaches to construct the navigation-helper agent, including fine-tuning a novel task-oriented multi-modal response generation model that can see and respond, named SeeRee, and employing a multi-modal large language model in a zero-shot manner. Analysis of the task and method was conducted based on both automatic benchmarking and human evaluations.

[Paper](https://arxiv.org/abs/2305.14260)

[Webpage](https://sites.google.com/view/response2helprequests/home)


## Data

In order to automatically
evaluate conversational multi-modal navigation helpers in a cooperative dynamic, we propose two tasks, Response to Dialog History (RDH) and Response during Interaction (RdI). **We first convert the three exsisting Vision-and-Dialog datasets to fit the input and output of RDH task,** where the three datasets are:
- CVDN with photo-realistic indoor visual environment.
- AVDN with photo-realistic ourdoor visual environment.
- DialFRED with sythetic indoor visual environment.

We format and convert these datasets to suit our goal of training and evaluating multimodal navigation-helper agents. Each data sample contains a natural language inquirey about the navigation from the task performer, visual observation from the task performer, a sequence of images showing oracle information for the navigation and a natural language response corresponding to the inquirey. 

Three converted datasets are available at https://drive.google.com/drive/folders/11Le4tX3A_tPePgpc31c7Acgv33OX9JDl?usp=share_link

Each dataset is split to train, seen and unseen validation set according to the original splits. **Especially, we create a sample set for each dataset to help better understanding of our data**.

In RDH task, the helper agent outputs responses based on individual task performer's help requests among three different environments. We format and convert all three datasets to suit the training and evaluation based on RDH task. Each data sample contains the input as a natural language inquirey about the navigation from the task performer, visual observation from the task performer and a sequence of images showing oracle information for the navigation; the output as a natural language response corresponding to the inquirey. 

For RdI task, the helper agent needs to interact with the task performer consistently. Therefore the input data are sampled in real-time from the simulator without the need of any offline dataset, except some task definitions i.e. trajectory starting points and target positions.



## Code 

We demonstrate how to use R2H benchmark based on our helper agent, SeeRee.

SeeRee takes vision and language inputs and outputs the natural language response correspond to the inquirey in the input. The language input is history dialog (between the helper and task performer) and latest language inquirey from the task performer. The vision input is the latest task performer's observation and the observation along the future ground truth trajectory to the navigation target. Please follow the following to train and evaluate SeeRee. The training and evaluation are based on the data with RDH format we shared above.

**Prerequisite**

 * We recommand using the docker envirionment provided by [SwinBERT](https://github.com/microsoft/SwinBERT#before-running-code-launch-docker-container) to run our code. We will later provide a non-docker envirionment setup tutorial. 

 * [Weight download](https://drive.google.com/drive/folders/1hQqS9WJF9u0YmTVOb4VyBFho3TLT4pzl?usp=sharing)
   * Weights of Video Swin model. 
   * evalcap.

 * (optional) You may download the weight of SeeRee trained by us and skip training it by yourself: [TODO]

### Script for train (for either task):

```./SeeRee/scripts/train_seeree.sh```

### Script for eval (on RDH task):

By running the evaluation script, we get a coco format json file containing the predicted responses for every trajectories in the validation set. 

```./SeeRee/scripts/eval_seeree.sh```

We provide the raw evaluation outputs (coco format) that we used for the experiment result in our paper [here](https://drive.google.com/drive/folders/1Adjwyj2l7sYxJ0W4Mf7WS0QKodTdNBfN?usp=sharing). 

Then, we replace the original human responses in the CVDN/AVDN/DialFRED validation set with the predicted responses from SeeRee and run evaluation of the task performer on this modified validation set. 

### Script for real-time inference (for RdI task):

In RdI task, since the helper agent needs to interact with the task performer consistently, we deploy both the helper agent and task performer agent in the Matterport3D simulator. With the script below, SeeRee will run as a real-time api for responding to any help request. 

```./SeeRee/scripts/online_inference_seeree.sh```

<br />
<br />


<br />
<br />

*Please cite our paper as below if you use our work.*
```
@misc{fan2023r2h,
      title={R2H: Building Multimodal Navigation Helpers that Respond to Help Requests}, 
      author={Yue Fan and Jing Gu and Kaizhi Zheng and Xin Eric Wang},
      year={2023},
      eprint={2305.14260},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

We build our code based on [SwinBERT](https://github.com/microsoft/SwinBERT/tree/main). Our code is release with the under MIT license.