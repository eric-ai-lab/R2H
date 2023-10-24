# R2H: Building Multimodal Navigation Helpers that Respond to Help Requests

## Data
Our R2H benchmark is built upon three exsisting Vision-and-Dialog datasets:
- CVDN with photo-realistic indoor visual environment.
- AVDN with photo-realistic ourdoor visual environment.
- DialFRED with sythetic indoor visual environment.

We format and convert these datasets to suit our goal of training and evaluating multimodal navigation-helper agents. Each data sample contains a natural language inquirey about the navigation from the task performer, visual observation from the task performer, a sequence of images showing oracle information for the navigation and a natural language response corresponding to the inquirey. 


## Code

RDH task
In this task, helper agent output responses based on individual task performer's requests. 

The input, a set of task performer's requests can be downloaded here:

Then we can run the helper agent (in this repo, SeeRee) on the task performer's requests:

