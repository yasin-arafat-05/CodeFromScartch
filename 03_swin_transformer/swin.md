
<br>
<br>

# `# Swin Transformer:`

<br>
<br>

**S -> Shifted** <br>
**Win -> Window** <br>
**Transformer -> Transformer** <br>

- #01: Introduction
- #02: Patch Partition VS Linear Embedding VS Patch Merging
- #03: Hierarchical Representation
- #04: Swin Transformer Block
- #05: Complexity Calculation in ViT
- #06: Window Base Multihead-Self-Attention
- #07: Patching
- #08: Complexity in W-MSA

<br>

# `#01: Introduction:`

<br>

An overview of the Swin Transformer architecture is presented in Figure 3, which illustrates the tiny version (SwinT). It first splits an input RGB image into non-overlapping patches by a patch splitting module, like ViT. Each patch is
treated as a “token” and its feature is set as a concatenation of the raw pixel RGB values. In our implementation, we use a patch size of 4 × 4 and thus the feature dimension of each patch is 4 × 4 × 3 = 48. A linear embedding layer is applied on this raw-valued feature to project it to an arbitrary dimension (denoted as C).

`In Linear Projection we use, Conv2D Layer where, Kernel_size=(4x4) stride=(4,4), output Channel: (C->it's a hyperparameter, 96,192). Like,for image, [224,224,3] -> [56,56,C].`

`আমরা CNN দেখেছি, যদি image [6,6,3] হয় এতে convolution operation perform করলে, kernel_size =(3,3) এর জন্য তাহলে, convolution operation শেষ হলে  আমরা (4,4,3), 3 dimension feature map পাচ্ছি । যেইটা, আমাদের input image এর dimension এর সমান । এখন, প্রশ্ন হচ্ছে যে, Swin Transformer এ তো আমরা image এর channel dimension বাড়াচ্ছি, তো এইটা কীভাবে হচ্ছে? আর,  channel dimension বাড়ালে, লাভ কি?`

**Channel Dimension বাড়ালে লাভ:**
- Increase feature extraction power:
        কম চ্যানেল (3) শুধুমাত্র রঙ বা সাধারণ প্যাটার্ন ধরতে পারে। বেশি চ্যানেল (যেমন 96) মডেলকে জটিল ফিচার (যেমন OCT-এ রেটিনাল অস্বাভাবিকতা) শিখতে সাহায্য করে।
- মডেলের ক্ষমতা:
        বেশি চ্যানেলের মাধ্যমে মডেল আরও গভীর এবং বিস্তৃত হয়, যা OCT-এর মতো জটিল ডেটায় ভালো কাজ করে।
- Fexibility:
        C-কে টিউন করা যায়, যা বিভিন্ন ডেটাসেটের জন্য মডেলকে উপযোগী করে তোলে।

**How to increase channel dimension:**

`আমরা already CNN এ দেখে এসেছি, RGB image  এর ক্ষেত্রে আমরা kernel 3 টা ব্যবহার করি। তাই, আমরা output feature map এ channel 3 dim এর হয় । যদি 3 টা ব্যবহার করে আরো বেশি করি তাহলে?? হ্যা এইখানে, একই কাজ করা হয় । `

![image](img/img01.png)

<br>

# `#02: Patch Partition VS Linear Embedding VS Patch Merging`

<br>

```
Input Image [224, 224, 3]
    ↓
Patch Partition → [56, 56, 48] **(4×4 patches, each with 48 features)**
    ↓
Linear Embedding → [56, 56, C] **(Projected to C channels, e.g., C=96)**
    ↓
Swin Transformer Block → [56, 56, C] **(Feature transformation, token number unchanged)**
    ↓
Patch Merging → [28, 28, 2C] **(2×2 patch merge, reduces tokens by 4, channel to 2C)**
    ↓
Swin Transformer Block → [28, 28, 2C] **(Feature transformation)**
    ↓
Patch Merging → [14, 14, 4C] **(2×2 patch merge, reduces tokens by 4, channel to 4C)**
    ↓
Swin Transformer Block → [14, 14, 4C] **(Feature transformation)**
```
<br>

# `#03: Hierarchical Representation:`

<br>

**Hierarchical representation** refers to a multi-level structure of features or data where information is organized and processed at various scales or levels of abstraction. In the context of deep learning, particularly in models like the Swin Transformer, it involves extracting low-level details (e.g., edges or textures) at initial stages`(শুরুতে, patch গুলো অনেক  ছোট থাকে তখন, আমরা low-level details গুলো পাই । যখন, এইটা, আস্তে আস্তে বড় হতে থাকে reducing spatial resolution), তখন, overall image এর details পাই । )` and progressively building higher-level abstractions (e.g., patterns or global context) as the network deepens. This is achieved by reducing the spatial resolution (e.g., through patch merging`patch কে merge করা`) while increasing the feature channel depth, allowing the model to capture both local and global information in a layered manner.


To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of 2 × 2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features. This reduces the number of tokens by a multiple
of 2×2 = 4 (2× downsampling of resolution), and the output dimension is set to 2C.**Like,** 

![image](img/img03.png)

Something,like we read *max-pooling and average-pooling* in CNN which reduce resolution.And swin transformer is simillar to RegNet Architecture.

<br>

# `#04 Swin Transformer Block:`

<br>

![img](img/img04.png)

`Instead of using Multihead-Self-Attention(MSA) we use Window-Multihead-Self-Attention(W-MSA) and Sifted-Window-Multihead-Self-Attention(SW-MSA).Why W-MSA or SW-MSA?Because in ViT or MSA, we calculate attention score one patch to all the other patches and it's computationaly expensive.`

![img](img/img05.png)


<br>

# `#05 Complexity Calculation in ViT:`

<br>

![img](img/img04.jpg)

![img](img/img05.jpg)

![img](img/img06.jpg)

`The complexity is to high. To reduce complexity we use Window base Multihead-Self Attention.`


<br>

# `#06 Window Base Multihead-Self-Attention:`

<br>

![img](img/img07.png)

`To reduce complexity, we use W-MSA. Where, we divide the whole image like above. But there is a problem, in each window we have relationship among all the patches. But, there also have relationship between one window to another as shown in picture. How we compute this?? To solve this problem we use SW-MSA.`

![image](img/img08.png)

`To capture the relationship among window, we can shift our at top most left corner. Then, do zero padding where don't have the original image. We need to do this for also top most right corner, bottom left corner and bottom right corner.So, in total we have 9 windows and our each window denoted by **M** `

![image](img/img09.png)

`But in this apporch, there is an problem. We need to do a lot of padding <PAD>. Like,from NLP Transformer, we use masking to avoid calculating zero padding. Because, there is no relationship among zero padding and all the sentence token.Here, we face the same problem we don't any relationship between zero Padding and our image. So, we don't use this process in swin Transformer. Insted of this, `

![image](img/img10.png)

`We divide the whole image and then shifted the image like shown in the above. This is call Cyclic Shift. After cyclic shift we divide the image into window.`

![image](img/img11.png)

`কিন্তু, সমস্যা হচ্ছে যে, D and B এর ক্ষেত্রে যদি আমরা self-attention apply করি । তাহলে, shifted image এর D এর শেষ প্রান্ত এর সাথে B এর প্রথম প্রান্ত মিলিত হয়েছে । কিন্তু, orginal image এ,  B এর শেষ প্রান্ত এর সাথে D এর প্রথম প্রান্ত মিলিত হয়েছে । তাই, যদি, shifted image এর D and B এর মধ্যে relationship বের করি তাহলে, এইটা কোন কাজে আসবে না কারণ, orginal image এ এদের কোন relation নেই । তাই, আমরা, এখানে, self-attention এর পরিবর্তে masked-self-attention apply করবো । যেখানে, masked-self-attention এ যদি orginal image এ relation থাকে তাহলে attention score বের করবো না থাকলে করবো না ।`

![image](img/img12.png)

`After appling masked-self-attention, we will do Reverse Cyclic Shift and made the image like orginal one.`

![image](img/img13.png)

`আমরা এই উপরের কাজ গুলো, swin-transformer-block করে থাকি । উপরে আমরা কোন swin-transformer-block এ কত গুলো  (two successive swin transformer block) ব্যবহার করতেছি সেইটা  বলে দেওয়া আছে । `



<br>

# `#07 Patching:`

<br>


![image](img/img14.png)


<br>

# `#08 Complexity in W-MSA:`

<br>

- **Ω(W-MSA) = 4hwC² + 2M²hwC**

**Here:**
- `h` = height in patches
- `w` = width in patches
- `C` = dimention of head
- `hw` = total token number
- `M` = window size in patches

**Ω(W-MSA) = 4hwC² + 2M²hwC**
**Ω(MSA) = 4hwC² + 2(hw)²C**
- MSA: 2(hw)²C

    - এটি গ্লোবাল অ্যাটেনশনের কোয়াড্রাটিক খরচ, যেখানে প্রতিটি টোকেন সব টোকেনের সাথে সম্পর্ক গণনা করে।

- W-MSA: 2M²hwC

    - এটি উইন্ডো-ভিত্তিক অ্যাটেনশনের লিনিয়ার খরচ, যেখানে অ্যাটেনশন শুধু M × M প্যাচের মধ্যে সীমাবদ্ধ।

