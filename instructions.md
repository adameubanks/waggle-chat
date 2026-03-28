Help me with the following project:


Deep Learning-Based Motion Tracking in Honeybee Hive Videos
Goal: This project aims to build a deep learning pipeline that detects and tracks individual
honeybees in hive video and estimates the orientation of their waggle motion from observed
trajectories. Waggle direction encodes information about food source location, making accurate
motion estimation biologically meaningful. It’s like having google translate for bees.

**Biology note (von Frisch):** In the waggle dance, **run angle (vs. gravity)** encodes the
**direction of the food source** relative to the sun/hive reference, and **waggle phase duration**
encodes **foraging distance** (longer run ≈ farther). So a useful decoded “vector” is
**(bearing, distance proxy)** — not angle alone.

Given short video clips, the system will:
● Detect and track bees across frames.
● **Primary focus here:** reliable **waggle detection** and **duration** (distance proxy);
  bearing can be estimated later with **classical CV** (e.g. elongated bee ROI / optical flow)
  and checked against annotations.
● The output will include **time-localized waggle segments** and **duration estimates**; angle
  can be added as a separate validated stage.
Dataset: The dataset consists of high-resolution hive videos stored on my lab’s supercomputer as filtered_waggles_n.csv and WaggleDance_n.mp4. The footage contains multiple bees moving simultaneously in natural hive
conditions. Some clips include annotated waggle segments or trajectory information; others may
require semi-supervised labeling.
Due to lack of data and computational expensiveness of video, I will split our dataset 80%
training and 20% validation/test.
Implementation Ideas:
1. Detection and Tracking
a. Use a CNN-based detector (e.g., finetuned ResNet) to detect bees per frame.
2. Orientation Estimation: CNN + LSTM or 3D CNN:
a. CNN extracts frame-level features.
b. LSTM models temporal motion patterns.
Training will use Adam or SGD with momentum. Loss function: Mean Squared Error (MSE).
Measures of Success: **Waggle detection** (precision/recall/F1 on time segments), **duration**
error (MAE/RMSE in seconds vs annotations). **Angle** validated separately (e.g. OpenCV on bee ROI)
against labelled bearing. Mean Absolute Angular Error (MAE) in degrees where angle is modelled.
I will assess generalization based on validation/test set metrics.
Computational Resources: Training will be conducted on BYU’s supercomputing cluster. Data
is accessible in the directory as filtered_waggles_n.csv and WaggleDance_n.mp4. Implementation will use PyTorch with GPU acceleration

Dataset Description
Dataset
The dataset consists of video recordings of honeybee activity inside a hive, captured under natural
conditions. The footage is stored in MP4 format at 720p resolution and includes three clips totaling
about one hour of video. The videos contain multiple bees moving simultaneously, including examples
of waggle dance behavior. The data is stored on filtered_waggles_n.csv and WaggleDance_n.mp4. I accessed the files
by SSHing into the lab machine and transferred them to BYU’s High Performance Computing (HPC)
cluster, where the analysis and model training will be performed.
Preprocessing
Because the raw videos are long and computationally expensive to process, they will be divided into
smaller segments. Each clip will be split into 3■second windows, which is approximately the duration of
a waggle dance.
• extract frames from each segment
• resize frames to a consistent format
• remove corrupted segments
• align segments with any existing waggle annotations or add semi■supervised labels
Train/Test Split
The dataset will be split 80% for training and 20% for validation/testing due to the limited amount of
available video data.

Submit a PDF writeup of your project, no more than 3 pages in length. It must include

A brief description of the problem you have tackled, the data you used, your technical approach and your results (no more than 2 pages)
An accounting that shows the total amount of time you spent on your final, broken down by day (no more than one page)
 

Grading rubric
Your final project counts as 25% of your overall grade.

Grading is divided into two parts:

20% number of hours you spent
5% report quality
 

For the number of hours, I will take the total number of hours and divide by 30. This will be your percentage.

I will evaluate your report based on the quality of both your project and your writing.

 

Note that this project is your final exam.

 

Description
For your final project, you should execute a substantial project of your own choosing. You will turn in a report (in PDF format). Your writeup can be structured in whatever way makes sense for your project (see below for some possible outlines), and it must be polished, well-written and generally of high quality.

Your project will be graded more on effort than results. As I have stated in class, I would rather have you swing for the fences and miss, than take on a simple, safe project. It is therefore very important that your final time log clearly convey the scope of your efforts.

I am expecting some serious effort on this project, and I am expecting that your report clearly conveys that. 

Important: your project must include some aspect of training (or finetuning) a model. Simply pulling a model from hugging face for inference, or calling an API (e.g., OpenAI, Gemini etc...) is not sufficient.

Requirements for the time log
For the time log, you must document the time you spent (on a daily basis) along with a simple description of your activities during that time. If you do not document your time, it will not count. In other words, it is not acceptable to claim that you spent 30 hours on your project, without a time log to back it up. I will not make any exceptions to this requirement.

So, for example, a time log might look like the following:

8/11 - 1 hour - read alphago paper
8/12 - 2 hours - downloaded and cleaned data
8/21 - 4 hours - found alphago code
8/24 - 1 hour - implemented game logic
9/17 - 2 hours - worked on self-play engine
9/18 - 1 hour - worked on self-play engine
10/1 - 2 hours - debugged initial training results
… etc.

 

Additional requirements:

You may not count any more than 5 hours of research and reading
You may not count any more than 10 hours of “prep work”. This could include dataset preparation, collection and cleaning; or wrestling with getting a simulator / model working for a deep RL project; etc.
At least 20 hours must involve designing, building, debugging and testing deep learning-based models, analyzing results, experimenting, etc. You may not include hours your model spends on training.
You don't get extra credit for more than 30 hours. Sorry. :)
 

Requirements for the report
Your writeup serves to inform me about what you did, and simply needs to describe what you did for your project. You should describe:

The problem you set out to solve
The exploratory data analysis you did
Your technical approach
Your results
It should be 1-2 pages.

 

Possible project ideas
Many different kinds of final projects are possible. A few examples include:

Learning how to render a scene based on examples of position and lighting
Learning which way is “up” in a photo (useful for drone odometry)
Training an HTTP server to predict which web pages a user will likely visit next
Training an earthquake predictor
Using GANs to turn rendered faces into something more realistic (avoiding the “uncanny valley”)
Transforming Minecraft into a more realistic looking game with DNN post-processing
Using style transfer on a network trained for facial recognition (to identify and accentuate facial characteristics)
Using RGB+Depth datasets to improve geometric plausibility of GANs
 

The project can involve any application area, but the core challenge must be tackled using some sort of deep learning.

The best projects involve a new, substantive idea and novel dataset. It may also be acceptable to use vanilla DNN techniques on a novel dataset, as long as you demonstrate significant effort in the “science” of the project – evaluating results, exploring topologies, thinking hard about how to train, and careful test/training evaluation. It may also be acceptable to simply implement a state-of-the-art method from the literature, but clear such projects with me first.

Notes
You are welcome to use any publicly available code on the internet to help you.

Here are some questions that you should answer as part of your report:

A discussion of the dataset
Where did it come from? Who published it?
Who cares about this data?
A discussion of the problem to be solved
Is this a classification problem? A regression problem?
Is it supervised? Unsupervised?
What sort of background knowledge do you have that you could bring to bear on this problem?
What other approaches have been tried? How did they fare?
A discussion of your exploration of the dataset
Before you start coding, you should look at the data. What does it include? What patterns do you see?
Any visualizations about the data you deem relevant
A clear, technical description of your approach
Background on the approach
Description of the model you use
Description of the inference / training algorithm you use
Description of how you partitioned your data into a test/training split
How many parameters does your model have? What optimizer did you use?
What topology did you choose, and why?
Did you use any pre-trained weights? Where did they come from?
An analysis of how your approach worked on the dataset
What was your final RMSE on your private test/training split?
Did you overfit? How do you know?
Was your first algorithm the one you ultimately used for your submission? Why did you (or didn't you) iterate your design?
Did you solve (or make any progress on) the problem you set out to solve?
 

Possible sources of interesting datasets
Figure Eight (formerly CrowdFlower)
KDD cup
UCI repository
CVonline
Kaggle (current and past)
Data.gov
AWS
World Bank
BYU CS478 datasets
data.utah.gov
Google research
BYU DSC competition
Make your own...
