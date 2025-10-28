# New Dataset

## HMDB51 Dataset

### Download the dataset

using curl

```bash
curl -L -o hmdb51.zip https://huggingface.co/datasets/hibana2077/sample-action-reg-data/resolve/main/hmdb51.zip?download=true
unzip hmdb51.zip -d ./datasets/hmdb51 # if you don't want to see unzip logs, use `unzip -q`
```

### Dataset Structure

```
hmdb51/
├── train/
│   ├── metadata.csv
│   ├── stand_TheBoondockSaints_stand_f_cm_np1_fr_med_59.mp4
│   ├── stand_TheBoondockSaints_stand_u_cm_np1_fr_bad_45.mp4
│   └── ...
├── validation/
│   ├── metadata.csv
│   ├── walk_Crash_walk_f_cm_np1_le_med_7.mp4
│   ├── walk_Crash_walk_u_nm_np1_ba_med_3.mp4
│   └── ...
├── test/
│   ├── metadata.csv
│   ├── smoke_smoking_smoke_h_nm_np1_fr_med_0.mp4
│   ├── smoke_you_like_a_nice_long_white_cig_smoke_u_nm_np1_ri_goo_2.mp4
│   └── ...
```

### Metadata Format

#### Train/metadata.csv

```
video_id,file_name,label,original_filename,duration,fps,frame_count,width,height,resolution,file_size_original,file_size_mp4,split
eat_Crash_eat_h_cm_np1_fr_med_11,eat_Crash_eat_h_cm_np1_fr_med_11.mp4,eat,Crash_eat_h_cm_np1_fr_med_11.avi,2.6333333333333333,30.0,79,560,240,560x240,266752,167596,train
```

## Diving Dataset

### Download the dataset

Using curl

```bash
curl -L -o Diving48_rgb.tar.gz https://huggingface.co/datasets/bkprocovid19/diving48/resolve/main/Diving48_rgb.tar.gz?download=true
tar -xzvf Diving48_rgb.tar.gz # output folder rgb
```

Need to using label file from:

- "../src/core/constant/Div48/Diving48_V2_test.json"
- "../src/core/constant/Div48/Diving48_V2_train.json"
- "../src/core/constant/Div48/Diving48_vocab.json"

### Dataset Structure

```
rgb/
├── 2x00lRzlTVQ_00000.mp4
├── 2x00lRzlTVQ_00001.mp4
└── ...
```

### Label Format

#### Diving48_V2_train.json

```json
[
  {
    "vid_name": "-mmq0PT-u8k_00155",
    "label": 0,
    "start_frame": 0,
    "end_frame": 48
  },
  {
    "vid_name": "-mmq0PT-u8k_00156",
    "label": 0,
    "start_frame": 0,
    "end_frame": 70
  },
    ...
]
```

#### Diving48_vocab.json

```json
[
  [
    "Back", 
    "15som", 
    "05Twis", 
    "FREE"
  ], 
  [
    "Back", 
    "15som", 
    "15Twis", 
    "FREE"
  ], 
  [
    "Back", 
    "15som", 
    "25Twis", 
    "FREE"
  ],
    ...
]
```

## Something-Something V2 Dataset

### Download the dataset

Using download script from [here](../datasets/ssv2.bash)

```bash
mkdir -p ./datasets/ssv2
cp download/ssv2.bash ./datasets/ssv2/
cd ./datasets/ssv2
bash ssv2.bash
```

### Dataset Structure

```
ssv2/
├── 20bn-something-something-v2/
│   ├── 1.webm
│   ├── 10.webm
│   └── ...
├── labels/
│   ├── labels.json
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   └── test-answers.csv
```

### Label Format

#### labels/labels.json

```json
{
 "Approaching something with your camera":"0",
 "Attaching something to something":"1",
 "Bending something so that it deforms":"2",
...
  "Wiping something off of something":"173"
}
```

#### labels/train.json

```json
[
{"id":"78687","label":"holding potato next to vicks vaporub bottle","template":"Holding [something] next to [something]","placeholders":["potato","vicks vaporub bottle"]},
{"id":"42326","label":"spreading margarine onto bread","template":"Spreading [something] onto [something]","placeholders":["margarine","bread"]},
...
{"id":"145274","label":"moving ladle away from hot pack","template":"Moving [something] away from [something]","placeholders":["ladle","hot pack"]},
{"id":"131791","label":"pretending to put vessel onto poori maker","template":"Pretending to put [something] onto [something]","placeholders":["vessel","poori maker"]}
]
```

#### labels/validation.json

```json
[
{"id":"74225","label":"spinning cube that quickly stops spinning","template":"Spinning [something] that quickly stops spinning","placeholders":["cube"]},
{"id":"116154","label":"showing clay box on top of wallet","template":"Showing [something] on top of [something]","placeholders":["clay box","wallet"]},
...
{"id":"117478","label":"moving glass up","template":"Moving [something] up","placeholders":["glass"]},
{"id":"36585","label":"plastic falling like a feather or paper","template":"[Something] falling like a feather or paper","placeholders":["plastic"]}
]
```

#### labels/test.json

```json
[
{"id":"1420"},
{"id":"166429"},
...
{"id":"211642"},
{"id":"215493"}
]
```

#### labels/test-answers.csv

```csv
1420;Throwing something
50058;Stacking number of something
...
211642;Scooping something up with something
215493;Pretending to poke something
```

## Breakfast Dataset

### Download the dataset

Using gdown

```bash
gdown 1jgSoof1AatiDRpGY091qd4TEKF-BUt6I
tar -xzf BreakfastII_15fps_qvga_sync.tar.gz
mv BreakfastII_15fps_qvga_sync breakfast
```

### Dataset Structure

```
breakfast/
├── P03/
│   ├── cam01/
│   │   ├── P03_cereals.avi
│   │   ├── P03_cereals.avi.labels
│   │   └── ...
│   ├── webcam01/
│   │   ├── P03_cereals.avi
│   │   ├── P03_cereals.avi.labels
│   │   └── ...
│   └── webcam02/
│       ├── P03_cereals.avi
│       ├── P03_cereals.avi.labels
│       └── ...
├── ...
├── P54/
│   ├── cam01/
│   │   ├── P54_cereals.avi
│   │   ├── P54_cereals.avi.labels
│   │   └── ...
│   ├── webcam01/
│   │   ├── P54_cereals.avi
│   │   ├── P54_cereals.avi.labels
│   │   └── ...
│   └── webcam02/
│       ├── P54_cereals.avi
│       ├── P54_cereals.avi.labels
│       └── ...
```

### Split

- Part 1: P03 - P15
- Part 2: P16 - P28
- Part 3: P29 - P41
- Part 4: P42 - P54

Train on Part 1-3, Test on Part 4.

### Label Format

10 classes:

- Bowl of cereals
- Coffee
- Chocolate milk
- Orange juice
- Fried eggs
- Fruit salad
- Pancakes
- Scrambled eggs
- Sandwich
- Tea

## UCF101 Dataset

### Download the dataset

Using curl

```bash
curl -L -o UCF101.zip https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip?download=true
unzip -q UCF101.zip # if you don't want to see unzip logs, use `unzip -q`. I relly recommend using `unzip -q`
mv UCF-101 UCF101
```

also download the annotation files.

```bash
curl -L -o UCF101TrainTestSplits-RecognitionTask.zip https://huggingface.co/datasets/quchenyuan/UCF101-ZIP/resolve/main/UCF101TrainTestSplits-RecognitionTask.zip?download=true
unzip UCF101TrainTestSplits-RecognitionTask.zip # if you don't want to see unzip logs, use `unzip -q`. I relly recommend using `unzip -q`
mv ucfTrainTestlist UCF101/ucfTrainTestlist
```

### Dataset Structure

```
UCF101/
├── ApplyEyeMakeup/
│   ├── v_ApplyEyeMakeup_g01_c01.avi
│   ├── v_ApplyEyeMakeup_g01_c02.avi
│   ├── ...
│   ├── v_ApplyEyeMakeup_g02_c01.avi
│   └── ...
├── ...
├── YoYo/
│   ├── v_YoYo_g01_c01.avi
│   ├── v_YoYo_g01_c02.avi
│   ├── ...
│   ├── v_YoYo_g02_c01.avi
│   └── ...
├── ucfTrainTestlist/
│   ├── classInd.txt
│   ├── testlist01.txt
│   ├── testlist02.txt
│   ├── testlist03.txt
│   ├── trainlist01.txt
│   ├── trainlist02.txt
│   └── trainlist03.txt
```

### Label Format

#### ucfTrainTestlist/classInd.txt

```txt
1 ApplyEyeMakeup
2 ApplyLipstick
3 Archery
...
99 WallPushups
100 WritingOnBoard
101 YoYo
```

#### ucfTrainTestlist/trainlist01.txt

```txt
ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi 1
ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c02.avi 1
...
YoYo/v_YoYo_g25_c04.avi 101
YoYo/v_YoYo_g25_c05.avi 101
```

#### ucfTrainTestlist/testlist01.txt

```txt
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi
ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02.avi
...
YoYo/v_YoYo_g07_c03.avi
YoYo/v_YoYo_g07_c04.avi
```