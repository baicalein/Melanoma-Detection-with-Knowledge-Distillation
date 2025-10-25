**Milestone I: Literature Review & Proposal**\
**Lightweight Melanoma Classification: Knowledge Distillation**

**Motivation**\
Smartphone apps for melanoma detection are becoming more popular. Two
groups especially show interest. First, people with a personal or family
history of skin cancer. Second, patients who take long-term
immune-suppressing drugs like TNF blockers, which carry FDA warnings
about cancer risk. The second group, in particular, often feels anxious
and wants early detection.

While these apps are convenient, we wonder what the app score really
means and how often it fails to detect melanoma. These important
measures are hard to know and usually not shared. Because of this, users
may not trust the app, especially those who cannot afford to miss a
diagnosis.

This project will create a small and efficient model that works well on
smartphones. It will give results at one clear and important point on
the sensitivity-specificity curve so that people can understand the
trade-off.

To make the model run on a phone without needing a server, we will use
knowledge distillation. This is a model compression method that is often
seen as more promising than pruning or quantization. A large teacher
model will train a smaller student model. The smaller model is what we
will use. This lets us keep the model size small while still focusing on
what matters most here: detecting melanoma reliably with high
sensitivity and clear, honest reporting.

**Dataset**\
HAM10000\
[[https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000]{.underline}](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

**Literature Review**

**Previous Methods**\
Recent studies on dermoscopy images use knowledge distillation to teach
a small student model by copying a stronger teacher model. This is done
by combining regular classification loss with a second loss that uses
the teacher's softened prediction scores. The teacher model stays
frozen. Only the student is trained and later used. This method, which
includes softmax temperature and hard/soft loss mixing, is already
described in skin lesion studies.

One study by Islam et al. used a large teacher made by combining
ResNet152V2, ConvNeXt-Base, and ViT-Base. This model had about 236
million parameters. It trained a small student model with three blocks,
which was later compressed. They found that global max pooling worked
better than global average pooling for small models. They also reported
that simple KD settings (temperature 1 and mixing alpha 0.5) worked well
on HAM10000.

Another study by Kabir et al. used smaller teachers like ResNet50 and
DenseNet161. They trained a student with fewer than 1 million
parameters, using the same softened loss method. They showed that even
very small models can still perform well on HAM10000.

In addition, Nazari et al. proposed compact attention‑augmented
EfficientNet‑B3 models for melanoma detection, combining high accuracy
with up to 98% fewer parameters and 20× faster inference than large ISIC
challenge winners, demonstrating their suitability for mobile
deployment.

**Gaps in Current Research**\
In most KD studies, results are reported using overall accuracy, F1
score, or parameter size. But for patients and healthcare providers,
what matters most is how well the model detects melanoma at one clear
operating point. This means they want to know what the precision and
specificity are when sensitivity is very high. Current KD studies do not
focus on this. The binary and multiclass studies do not report results
that center on melanoma. The segmentation study does not report
classification thresholds.

In our project, we will create a small classifier and focus the
evaluation on melanoma-specific operating points that are easy to
understand.

**Proposed Method (Initial Plan)**

**Teacher and Student Models**\
We will use ResNet50 as the teacher. It will start with pre-trained
weights from ImageNet and will use a basic global average pooling head.
We will freeze the teacher and fine-tune it. The student will be
MobileNetV3-Small with a lightweight head. Training will use KD. The
loss will combine a hard-label binary cross-entropy loss and a soft loss
between teacher and student outputs. The teacher will stay frozen.

**Handling Class Imbalance and Head Variants**\
We will start with weighted binary cross-entropy, using weights based on
the training data. We will test both BCE and focal loss, and we will try
both global max pooling and global average pooling in the student head.
The earlier binary KD study showed that GMP was better for small models.

**KD Hyperparameters**\
We will run a small test using temperature values of 1 and 2 and mixing
values of 0.1, 0.5, and 0.9. This will help us see how strong the soft
targets should be when using BCE or focal loss, and how it interacts
with the type of pooling head. Some earlier work showed that temperature
1 worked best on HAM10000, but multiclass studies found different best
values. We will check in our binary setup.

**Model Size and Compute**\
We will report the number of parameters and FLOPs for each ablation to
show the compute cost. We will not report phone or CPU timing yet. That
will come in a later milestone.

**Experiments and Evaluation Plan**

**Data and Splits**\
We will use HAM10000 with binary labels (melanoma vs. non-melanoma).
Splits will be done to avoid data leakage. We will either use 5-fold
cross-validation or a fixed train, validation, and test split. We will
log the random seeds used.

**Main Metrics (Focus on Operating Point)**\
We will report ROC-AUC and PR-AUC for melanoma as the positive class.\
We will choose a threshold using the validation set to get 95%
sensitivity. Then we will report test specificity, positive predictive
value, and negative predictive value at that threshold.

**Ablation Studies**

- Loss: compare weighted BCE and focal loss with gamma values of 1, 2,
  and 4 under KD

- Head: compare global max pooling and global average pooling in the
  student

- KD settings: test temperature values of 1 and 2, and mixing values of
  0.1, 0.5, and 0.9

**Reporting**\
For each ablation, we will report AUC scores, performance at the 95%
sensitivity point (specificity, PPV, NPV), number of parameters, and
FLOPs. We will not include latency in this milestone. If needed, we can
add device timing in a later milestone.

**Reference**

1.  Islam, N., Hasib, K. M., Joti, F. A., Karim, A., & Azam, S. (2024).
    Leveraging Knowledge Distillation for Lightweight Skin Cancer
    Classification: Balancing Accuracy and Computational Efficiency.
    arXiv preprint arXiv:2406.17051. <https://arxiv.org/abs/2406.17051>

2.  Kabir, M. R., Borshon, R. H., Wasi, M. K., Sultan, R. M., Hossain,
    A., & Khan, R. (2024). Skin cancer detection using lightweight model
    souping and ensembling knowledge distillation for memory-constrained
    devices*.* Intelligence-Based Medicine, 10, 100176.
    <https://doi.org/10.1016/j.ibmed.2024.100176>

3.  Nazari, S., & Garcia, R. (2025). Going smaller: Attention-based
    models for automated melanoma diagnosis. Computers in Biology and
    Medicine, 185*, 109492.*
    <https://www.sciencedirect.com/science/article/pii/S0010482524015774?via%3Dihub>
