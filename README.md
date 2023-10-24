# R2H: Building Multimodal Navigation Helpers that Respond to Help Requests


Todos:
- [x] SeeRee code released
- [x] RDH Data released
- [x] RDH sample code released
- [ ] RdI Data released
- [ ] RdI Code released


## Data
Our R2H benchmark is built upon three exsisting Vision-and-Dialog datasets:
- CVDN with photo-realistic indoor visual environment.
- AVDN with photo-realistic ourdoor visual environment.
- DialFRED with sythetic indoor visual environment.


In order to automatically
evaluate conversational multi-modal navigation helpers in a cooperative dynamic, we propose two tasks, Response from Dialog History (RDH) and Response during Interaction (RdI). 

In RDH task, the helper agent outputs responses based on individual task performer's help requests. We format and convert all three datasets to suit the training and evaluation on RDH task. Each data sample contains the input as a natural language inquirey about the navigation from the task performer, visual observation from the task performer and a sequence of images showing oracle information for the navigation; the output as a natural language response corresponding to the inquirey. 

Data is available at https://drive.google.com/drive/folders/11Le4tX3A_tPePgpc31c7Acgv33OX9JDl?usp=share_link

In RdI task, the helper agent needs to interact with the task performer consistantly. We deploy both the helper agent and task performer agent in simulations of CVDN for evaluation. Code will be released soon. 

## Code

RDH task
In this task, helper agent output responses based on individual task performer's requests. 

The input, a set of task performer's requests can be downloaded here:

Then we can run the helper agent (in this repo, SeeRee) on the task performer's requests:

