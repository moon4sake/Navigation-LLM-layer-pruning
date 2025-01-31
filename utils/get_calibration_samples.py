import random
import numpy as np
import torch

from datasets import load_dataset
from torch.utils.data.dataset import Dataset


def get_c4(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_bookcorpus(tokenizer, n_samples, seq_len):
    traindata = load_dataset(
        'evaluate/bookcorpus.py', split='train'  # bookcorpus
    )

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_medqa(tokenizer, n_samples, seq_len):
    # traindata = load_dataset(
    #     'GBaker/MedQA-USMLE-4-options-hf', 'GBaker--MedQA-USMLE-4-options-hf', data_files={'train': 'train.json'}, split='train'
    # )
    traindata = load_dataset("datasets/GBaker--MedQA-USMLE-4-options-hf", split='train')

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i]['sent1'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)


def get_medmcqa(tokenizer, n_samples, seq_len, use_exp=False):
    # traindata = load_dataset(
    #     'GBaker/MedQA-USMLE-4-options-hf', 'GBaker--MedQA-USMLE-4-options-hf', data_files={'train': 'train.json'}, split='train'
    # )
    traindata = load_dataset("datasets/openlifescienceai--medmcqa", split='train')

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            if use_exp:
                j, temp = i, traindata[i]['exp']
                while temp is None:
                    j += n_samples
                    temp = traindata[j]['exp']
                tokenized_sample = tokenizer(temp, return_tensors='pt')
            else:
                tokenized_sample = tokenizer(traindata[i]['question'], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_medmcqa_cot(tokenizer, n_samples, seq_len):

    traindata = ['To determine which vitamin is supplied only from animal sources, we need to evaluate each option based on its natural sources and availability in the diet: (a) Vitamin C: This vitamin is primarily found in fruits and vegetables, such as citrus fruits, strawberries, bell peppers, and broccoli. It is not sourced from animal products, making it widely available in plant-based foods. (b) Vitamin B7 (Biotin): Biotin is found in a variety of foods, including both animal and plant sources. It is present in eggs, milk, nuts, seeds, and certain vegetables. Therefore, it is not exclusively sourced from animal products. (c) Vitamin B12 (Cobalamin): Vitamin B12 is unique because it is synthesized by microorganisms and is naturally found in significant amounts only in animal-derived foods, such as meat, fish, eggs, and dairy products. Plant-based foods do not naturally contain Vitamin B12 unless they are fortified. This makes Vitamin B12 the only vitamin in this list that is supplied solely from animal sources. (d) Vitamin D: While Vitamin D can be obtained from animal sources like fatty fish and egg yolks, it can also be synthesized by the human body when the skin is exposed to sunlight. Additionally, some plant-based foods are fortified with Vitamin D. Therefore, it is not exclusively sourced from animal products. Based on this analysis, the correct answer is (c) Vitamin B12, as it is the only vitamin in the list that is naturally supplied solely from animal sources.',
                    'To determine the correct statement regarding the lag phase in bacterial growth, let\'s evaluate each option based on the characteristics of the lag phase: (a) Time taken to adapt in the new environment: This statement accurately describes the lag phase. During this phase, bacteria are not actively dividing but are instead adapting to the new environment. They synthesize necessary enzymes and metabolic intermediates to prepare for subsequent growth. This adaptation period is crucial for the bacteria to adjust to changes in nutrient availability, temperature, and other environmental factors. (b) Growth occurs exponentially: This statement is incorrect for the lag phase. Exponential growth occurs during the log (exponential) phase, which follows the lag phase. During the lag phase, there is no significant increase in cell numbers as the bacteria are not yet dividing. (c) The plateau in lag phase is due to cell death: This statement is incorrect. The plateau observed in the lag phase is not due to cell death but rather the lack of cell division as bacteria acclimate to their new environment. Cell death is not a characteristic of the lag phase. (d) It is the 2nd phase in bacterial growth curve: This statement is incorrect. The lag phase is the first phase in the bacterial growth curve, followed by the exponential (log) phase, stationary phase, and death phase. Based on this analysis, the correct answer is (a) Time taken to adapt in the new environment, as it accurately describes the primary function of the lag phase in bacterial growth.',
                    'To determine the primary pharmacokinetic change occurring in geriatric patients, we need to consider how aging affects the body\'s ability to process drugs. Pharmacokinetics involves the absorption, distribution, metabolism, and excretion of drugs. Let\'s evaluate each option: (a) Gastric absorption: While aging can affect gastric absorption due to changes in gastric pH and motility, this is not the most significant pharmacokinetic change in geriatric patients. The impact on drug absorption is generally less pronounced compared to other pharmacokinetic processes. (b) Liver metabolism: Aging can lead to a decrease in liver size and blood flow, potentially affecting the metabolism of drugs. However, the extent of this change varies and is not as consistently significant as changes in renal function. (c) Renal clearance: This is the most significant pharmacokinetic change in geriatric patients. As people age, there is a notable decline in kidney function, specifically in the glomerular filtration rate (GFR). After age 40, creatinine clearance decreases by approximately 8 mL/min/1.73 m² per decade. This reduction in renal clearance can lead to the accumulation of drugs in the body, increasing the risk of adverse effects and toxicity. Despite normal serum creatinine levels, which can be misleading due to decreased muscle mass, the actual renal function is often reduced. (d) Hypersensitivity: This option refers to an increased sensitivity to drugs, which is more related to pharmacodynamics rather than pharmacokinetics. It does not directly address the changes in drug processing by the body. Based on this analysis, the correct answer is (c) Renal clearance, as it represents the most significant pharmacokinetic change occurring in geriatric patients due to aging. This change affects the elimination of drugs and is crucial for adjusting drug dosages in older adults.',
                    'To determine which condition a per rectum (PR) examination is not useful for diagnosing, we need to understand the nature and location of each condition: (a) Anal fissure: An anal fissure is a small tear in the lining of the anus. A PR examination can help identify an anal fissure by allowing the examiner to feel for any tears or irregularities in the anal canal. However, it is often painful for the patient, and visual inspection is usually preferred. (b) Hemorrhoid: Hemorrhoids are swollen veins in the lower rectum and anus. A PR examination can help detect hemorrhoids by allowing the examiner to feel for swollen veins or lumps in the anal canal. (c) Pilonidal sinus: A pilonidal sinus is a tract or cavity located in the sacrococcygeal region, typically containing hair and debris. It is situated in the skin and subcutaneous tissue, not within the rectum or anal canal. Therefore, a PR examination is not useful for diagnosing a pilonidal sinus, as it is not located in the area accessible by this examination. (d) Rectal ulcer: A rectal ulcer is a lesion in the lining of the rectum. A PR examination can help identify a rectal ulcer by allowing the examiner to feel for any irregularities or lesions in the rectal lining. Based on this analysis, the correct answer is (c) Pilonidal sinus, as it is not located in the rectal or anal area where a PR examination would be effective. Instead, it is found in the skin and subcutaneous tissue of the sacrococcygeal region, making a PR examination irrelevant for its diagnosis.',
                    'To determine the normal waist-hip ratio (WHR) for females, we need to understand the standard cut-off points used to assess health risks associated with body fat distribution. The waist-hip ratio is calculated by dividing the circumference of the waist by the circumference of the hips. It is a useful indicator of fat distribution and potential health risks. According to the World Health Organization (WHO) guidelines: - A waist-hip ratio of 0.80 or below for women is considered normal and indicates a lower risk of metabolic complications. - A waist-hip ratio above 0.80 for women is associated with an increased risk of health issues, such as cardiovascular diseases and type 2 diabetes. Let\'s evaluate each option: (a) 0.7: This value is below the normal cut-off point for women, indicating a lower risk of metabolic complications. However, it is not the standard cut-off point for defining normal WHR. (b) 0.8: This is the standard cut-off point for women, indicating the upper limit of a normal waist-hip ratio. A WHR of 0.8 or below is considered normal and suggests a lower risk of metabolic complications. (c) 0.9: This value is above the normal cut-off point for women and indicates an increased risk of metabolic complications. (d) 1.0: This value is significantly above the normal cut-off point for women and indicates a substantially increased risk of metabolic complications. Based on this analysis, the correct answer is (b) 0.8, as it represents the normal waist-hip ratio cut-off point for females, indicating a lower risk of metabolic complications.',
                    'Nagler\'s reaction is a specific test used to identify the presence of Clostridium perfringens, a bacterium known for producing alpha-toxin, which is a type of lecithinase. This reaction is used to differentiate C. perfringens from other Clostridium species based on its ability to produce this toxin. Here\'s how Nagler\'s reaction works: - Medium Preparation: The bacterium is grown on a medium containing 6% agar, 5% Fildes peptic digest of sheep blood, and 20% human serum. - Antitoxin Application: The medium is divided into two halves. One half is treated with an antitoxin specific to the alpha-toxin produced by C. perfringens. - Observation: When C. perfringens is cultured on this medium, the colonies on the half without antitoxin will be surrounded by a zone of opacity due to the action of the alpha-toxin on the lecithin in the medium. On the half with the antitoxin, there will be no opacity around the colonies because the antitoxin neutralizes the alpha-toxin. This reaction is specific to Clostridium perfringens because it is the only Clostridium species that produces the alpha-toxin responsible for the opacity in the medium. Let\'s evaluate each option: (a) Clostridium tetani: This bacterium is known for producing tetanospasmin, a neurotoxin, and does not produce the alpha-toxin involved in Nagler\'s reaction. (b) Clostridium botulinum: This bacterium produces botulinum toxin, which is not involved in Nagler\'s reaction. (c) Clostridium perfringens: This bacterium produces the alpha-toxin that causes the opacity in Nagler\'s reaction, making it the correct answer. (d) Clostridium septicum: This bacterium does not produce the alpha-toxin involved in Nagler\'s reaction. Based on this analysis, the correct answer is (c) Clostridium perfringens, as it is the bacterium that shows Nagler\'s reaction due to its production of alpha-toxin.',
                    'To determine which condition is associated with hyperviscosity, we need to understand the nature of each condition and how it affects blood viscosity: (a) Cryoglobulinemia: This condition is characterized by the presence of cryoglobulins in the blood, which are proteins that precipitate at temperatures below 37°C and redissolve upon warming. The precipitation of these proteins can lead to increased blood viscosity, particularly when the blood is cooled, such as in peripheral circulation. This can cause hyperviscosity syndrome, which is a clinical condition characterized by increased blood thickness, leading to impaired circulation and various symptoms. (b) Multiple myeloma: This is a cancer of plasma cells that often leads to the production of a large amount of monoclonal immunoglobulins (paraproteins). These proteins can increase blood viscosity, and hyperviscosity syndrome can occur in some cases of multiple myeloma, especially when there is a high level of IgA or IgG. (c) MGUS (Monoclonal Gammopathy of Undetermined Significance): This is a condition where there is a presence of monoclonal protein in the blood, but it is usually at a lower level than in multiple myeloma. MGUS typically does not cause hyperviscosity because the levels of monoclonal protein are not high enough to significantly increase blood viscosity. (d) Lymphoma: This is a type of cancer that affects the lymphatic system. While certain types of lymphoma can produce monoclonal proteins, hyperviscosity is not a common feature of lymphoma unless there is significant production of monoclonal proteins, which is more typical of Waldenström\'s macroglobulinemia. While both cryoglobulinemia and multiple myeloma can lead to hyperviscosity, the question specifically asks for the condition where hyperviscosity is seen, and cryoglobulinemia is directly associated with hyperviscosity due to the nature of cryoglobulins precipitating and affecting blood flow. Therefore, the correct answer is (a) Cryoglobulinemia.',
                    'Chronic urethral obstruction, such as that caused by benign prostatic hyperplasia (BPH), can lead to significant changes in the urinary tract, including the kidneys. Let\'s evaluate the potential changes in kidney parenchyma due to such an obstruction: (a) Hyperplasia: This refers to an increase in the number of cells, leading to tissue enlargement. Hyperplasia is not typically associated with the kidney parenchyma in the context of urethral obstruction. (b) Hypertrophy: This refers to an increase in the size of cells, leading to tissue enlargement. While hypertrophy can occur in some organs in response to increased workload, it is not the primary change seen in kidney parenchyma due to urethral obstruction. (c) Atrophy: This refers to a decrease in the size and function of an organ or tissue. In the case of chronic urethral obstruction, the back pressure from urine that cannot be properly excreted leads to hydronephrosis, which is the dilation of the renal pelvis and calyces. Over time, this increased pressure causes progressive atrophy of the kidney parenchyma, as the tissue is compressed and damaged due to the obstruction. (d) Dysplasia: This refers to abnormal growth or development of cells within tissues or organs. Dysplasia is not typically associated with the kidney parenchyma in the context of urethral obstruction. Based on this analysis, the correct answer is (c) Atrophy. Chronic urethral obstruction due to benign prostatic hyperplasia can lead to hydronephrosis, which results in the progressive atrophy of the kidney parenchyma due to the obstruction of urine outflow.',
                    'To determine which option is not a surgical procedure for morbid obesity, we need to understand the common bariatric surgeries used to treat this condition. Bariatric surgery is aimed at achieving weight loss through restriction of food intake, malabsorption of nutrients, or a combination of both. Let\'s evaluate each option: (a) Adjustable gastric banding: This is a restrictive procedure where an adjustable band is placed around the upper part of the stomach to create a small pouch. This limits food intake and promotes a feeling of fullness with less food. (b) Biliopancreatic diversion: This is a malabsorptive procedure that involves removing a portion of the stomach and rerouting the intestines to reduce nutrient absorption. It is often combined with a duodenal switch. (c) Duodenal Switch: This procedure is a combination of restrictive and malabsorptive techniques. It involves removing a large portion of the stomach and rerouting the intestines, similar to biliopancreatic diversion, to limit food intake and nutrient absorption. (d) Roux en Y Duodenal Bypass: This option is not a recognized bariatric procedure. The correct term is "Roux-en-Y gastric bypass," which is a well-known bariatric surgery that combines both restrictive and malabsorptive elements. It involves creating a small stomach pouch and rerouting the small intestine to this pouch, bypassing a portion of the stomach and duodenum. Based on this analysis, the correct answer is (d) Roux en Y Duodenal Bypass, as it is not a recognized surgical option for morbid obesity. The correct procedure is Roux-en-Y gastric bypass, not "Roux en Y Duodenal Bypass."',
                    'In the scenario described, a patient presents 6 hours after a snake bite with mild local edema at the injury site, no detectable abnormalities on examination, and normal laboratory reports. The key considerations in managing a snake bite include assessing the severity of envenomation and determining whether the snake is venomous. Let\'s evaluate each management option: (a) Incision and suction: This is an outdated and not recommended practice for snake bites. It can cause more harm than good, including increased risk of infection and tissue damage. (b) Wait and watch: This is the most appropriate management in this scenario. Given the mild local edema, absence of systemic symptoms, and normal lab results, the patient does not show signs of significant envenomation. Observing the patient for 8-12 hours is recommended to monitor for any delayed symptoms of envenomation, especially if the snake cannot be positively identified as non-venomous. (c) Local subcutaneous antisnake venom: This is not a standard practice. Antivenom is typically administered intravenously and is reserved for cases with clear signs of systemic envenomation or severe local effects. (d) Intravenous antisnake venom: This is indicated in cases of significant envenomation, which may include systemic symptoms (e.g., coagulopathy, neurotoxicity) or severe local tissue damage. In this case, the absence of such symptoms suggests that antivenom is not necessary at this time. Based on the information provided, the most appropriate management is (b) Wait and watch. This approach allows for monitoring the patient for any delayed symptoms of envenomation while avoiding unnecessary interventions.',
                    ]

    tokenized_samples, history = [], []
    for _ in range(n_samples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            tokenized_sample = tokenizer(traindata[i], return_tensors='pt')
            if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                history.append(i)
                break
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i + seq_len])
    return torch.cat(tokenized_samples, dim=0)

def get_examples(dataset, tokenizer, n_samples, seq_len=128):
    if dataset == 'c4':
        return get_c4(tokenizer, n_samples, seq_len)
    elif dataset == 'bookcorpus':
        return get_bookcorpus(tokenizer, n_samples, seq_len)
    elif dataset == 'medqa':
        return get_medqa(tokenizer, n_samples, seq_len)
    elif dataset == 'medmcqa':
        return get_medmcqa(tokenizer, n_samples, seq_len)
    elif dataset == 'medmcqa_exp':
        return get_medmcqa(tokenizer, n_samples, seq_len, True)
    elif dataset == 'medmcqa_cot':
        return get_medmcqa_cot(tokenizer, n_samples, seq_len)
    # elif dataset == 'bookcorpus':
    #     return get_bookcorpus(tokenizer, n_samples, seq_len)
    else:
        raise NotImplementedError