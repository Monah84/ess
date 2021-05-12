import pickle
import pandas as pd
import swifter
import numpy as np
import multiprocessing
from tqdm import tqdm
import logging
import os

import scispacy
import spacy
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.pipeline import EntityRuler
from spacy.tokens import Doc, Span, Token

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

tqdm.pandas()

class AbbreviationRecognizer(object):
    """Example of a spaCy v2.3 pipeline component that sets entity annotations
    based on list of single or multiple-word company names. abbreviations are
    labelled as ORG and their spans are merged into one token. Additionally,
    ._.has_abbreviation and ._.is_abbreviation is set on the Doc/Span and Token
    respectively."""

    name = "custom_abbreviation_detector"  # component name, will show up in the pipeline

    def __init__(self, nlp, label="ABBREVIATION"):
        self.label = nlp.vocab.strings[label]  # get entity label ID
        self.abbreviations =["AAA","AAT","AAL","AAOx3","A/B","ab","ABC","ABCD","ABCs","ABCDs","ABCDEs","Abd","ABE","ABG","ABI","ABMT",
                            "Abn","ABO","ABPA","ABPI","ABVD","ABX","a.c.","AC","ACB","AC&BC","Acc","ACCU","ACD","ACDF","Ace","ACEI","ACh",
                            "AChE","ACL","ACLS","ACS","ACTH","ACU","ad.","AD","ADA","ADC","ADCC","ADD","ADH","ADHD","ADHR","ADLs","ad lib",
                            "adm","Adn","ADP","ad part. dolent","ADR","AED","AEM","AF","AFB","AFib","AFO","AFP","Ag","AGA","AGES criteria",
                            "AGN","a.h.","AHF","AHG","AHH","AHR","AI","AICD","AID","AIDS","AIH","AIHD","AIPD","AIIRB","AIN","AIS","aka","ALA",
                            "Alc","ALG","ALI","Alk phos","ALL","ALP","ALPS","ALS","ALT","altern. d.","AMA","Amb","AMC","AMI","AML","AMP",
                            "AMPA receptor","AMS","Amt","ANA","ANCA","ANDI","ANF","ANP","ANS","Ant","Anti-","ANTR","ANUG","A&O","A&Ox3","A&Ox4",
                            "AODM","AOM","a.p.","AP","A&P","APACHE II","APAP","APC","APD","APKD","APECED","APGAR","APH","APLS","APMPPE","applic.",
                            "APR","APS","APSAC","aPTT","aq.","aq. bull.","aq. calid.","aq. dist.","aq. gel.","AR","ARB","ARC","ARDS","ARF","Arg","ARM",
                            "AROM","ART","ARVC","ARVD","AS","ASA","ASAP","ASC","ASCAD","ASCUS","ASCVD","ASD","ASGUS","ASH","ASHD","ASIS","ASO","ASOT",
                            "Ass","AST","ASX","AT","ATA","ATB","ATCC","ATG","AT-III","ATN","ATNR","ATP","ATRA","ATS","AUC","aur.","aur. dextro.","aur. laev",
                            "aurist.","AV","AVM","AVR","A&W","A/W","Ax","AXR","AZT","Bx","Ba","BAC","BAL","BAO","BAT","BBB","BBB L","BBB R","BC","BCC","BCG",
                            "BCP","BCX","BBMF","BD","BDD","BDI","BE","BFP","BGAT","BGL","BIB","BIBA","BID","b.i.d.","Bilat eq","BiPAP","BiVAD","BK","BKA",
                            "bl.cult","bld","BLS","BM","BMC","BMD","BMI","BMP","BMR","BMT","BNO","BNP","BO","B/O","BOA","BOI","BOM","BOOP","BP","BPAD","BPD",
                            "BPH","BPM","BPPV","BR","BRA","BRAT","BRATY","BRB","BRBPR","BRCA1 (gene)","BRCA1 (protein)","BRCA2 (gene)","BRCA2 (protein)","BRP",
                            "BS","BSA","BSC","BSE","BSL","BSP","BT","BTL","BTP","BUN","BV","BVP","BZDs","C","C1","C2","C-section","CA","CABG","CABP","CAD",
                            "CADASIL","CAG","CAGS","cAMP","CAH","CAPD","Caps","CAT","Cath","CBC","CBC/DIFF","CBD","CBS","CC","CCCU","CCE","C/C/E","CCF",
                            "CCK","CCK-PZ","CCR","CCU","CD","CDH","CDP","CEA","CF","CFA","CFIDS","CFS","CFTR","cGMP","CGN","CH","CHD","ChE","CHEM-7","CHEM-20",
                            "CHF","CHO","Chol","CHT","CI","CICU","CIDP","CIN","Circ","CIS","CJD","CK","CKD","CKMB","CLL","CM","CMD","CME","CML","CMML","CMP",
                            "CMS","CMT","CMV","CN","CNS","C/O","CO","COAD","COCP","COLD","Comp","Conj","COPD","CO2","COX-1","COX-2","COX-3","CP","CPAP","CPC",
                            "CPE","CPK","CPKMB","CPP","CPR","CPT","CR","CrCl","Creat","CREST","CRF","CRH","CRI","Crit","CRP","CRT","CS","C/S","C&S","C-spine",
                            "CsA","CSF","CT","CTA","CTP","CTS","CTU","CTX","CV","CVA","CVAT","CVC","CVD","CVI","CVP","CVS","CVID","c/w","CWP","Cx","CXR","DS",
                            "D5","D5W","d","DA","DAEC","DALY","DBP","DBS","D&C","D/C","DCBE","DCIS","DCM","DD","DDD","DDx","D&E","DES","DEXA","DH","DHE","DHEA-S",
                            "DHF","DI","DIB","DIC","DIP","DiPerTe","Dis","Disch","DiTe","DIU","DJD","DKA","dl","DLE","DM","DMD","DNA","DNI","DNR","DNAR","DOA",
                            "DOB","DOE","DOSS","DP","DPH","DPL","DPT","DRT","DSA","DSD","Dsg","DSM","dsRNA","DT","DTA","DTaP","DTP","DTR","DTs","DU","DUB","DVT",
                            "DW","DX","DZ","E","EAC","EACA","EAEC","EAF","EBL","EBM","EBT","EBV","EC","ECF","ECG","ECM","ECHO","ECI","ECLS","ECMO","ECP","ECT",
                            "ED","EDC","EDD","EDH","EDM","EDRF","EDTA","EDV","EEE","EEG","EENT","EEX","EF","EFM","EGBUS","EGD","EGF","EHEC","EIEC","EJ","EKG",
                            "ELLSCS","ELISA","EmBx","EMC","EMD","EMF","EMG","EMLSCS","EMU","Emul","ENT","EOM","EOMI","EPEC","EPH","EPO","EPS","ER","ERCP","ESL",
                            "ESBL","ESR","ESRD","ESV","ESWL","ET","ETEC","Etiol","ETOH","ETS","ETT","EUA","EUP","EUS","EVAR","EVF","Exam","Exp Lap","Ext","Fx",
                            "FAMMM syndrome","FAP","FAST","FB","FBC","FBE","FBG","FBS","F/C","FDC","FDIU","FDP","FEV1","Fe","fem","FEP","FF","FFA","FFP","FHR",
                            "FHS","FHT","FHx","FIBD","FISH","FLK","fl.oz.","FMF","FMP","fMRI","F→N","FNA","FNAB","FNAC","FNC","FNH","FOBT","FOF","FOS","FPG",
                            "FROM","FSBS","FSE","FSH","FTA","FTA-ABS","FTT","F/U","FUO","FVC","FWB","FWD","G","G6PD","GA","GABA","GB","GBS","GBM","GC","GCA",
                            "GCS","G-CSF","GDA","GDLH","GDP","GERD","GFR","GGT","GGTP","GH","GHRF","GI","GIFT","GIST","GIT","GITS","GMC","GM-CSF","GMP","GN",
                            "GNRH","GOAT","GOD","GOK","GOMER","GORD","GOT","GP","GPT","Gr","GRAV I","GSW","GTN","GTT","Gtts","GU","GUM","GvH","GvHD","GYN","h",
                            "Hx","HA","H/A","HAA","HAART","HACE","HACEK","HAE","HAD","HALE","HAPE","HAV","Hb","Hb%","HbA","HbA1C","HBD","HbF","HBP","HbsAg",
                            "HBV","HC","HCC","hCG","HCL","HCM","Hct","HCRP","HCTZ","HD","HCV","HDL","HDL-C","HDU","HDV","H&E","HEENT","HELP","HELLP","HEMA",
                            "HES","HETE","HEV","HFM","HFRS","HGB","HGSIL","HGV","HGPRTase","HH","H&H","HHT","HHV","HI","Hib","HIDS","HIT","HIV","HL","HLA",
                            "HLHS","H&M","HMD","HMG-CoA","HMGR","HMS","HMSN","HN","HND","HNPCC","H/O","HOB","HOCM","HONK","H&P","HPA","HPETE","HPF","HPI",
                            "H/oPI","HPOA","HPL","HPS","HPV","HR","HRCT","HRT","h.s.","hs","H→S","HSC","HSG","HSIL","HSP","HSV","HT","HTLV","HTN","HTPA",
                            "HTVD","HUS","HVLT","131I or I131","IA","IABP","IAI","IBC","IBD","IBS","IC","ICD","ICDS","ICD-10","ICF","ICG","ICH","ICP","ICS","ICU",
                            "ICCU","I&D","IDA","IDC","IDDM","IDL","IDP","IF","IFG","Ig","IgA","IgD","IgE","IgG","IgM","IGT","IHC","IHD","IHSS","IM","IMA","IMB"
                            ,"IMI","IMN","IMT","IMV","Inc","INF(-α/-β/-γ)","INH","Inj","INR","Int","IO","I&O","IOL","IOP","IP","IPF","IPS","IPPB","IPPV","IQ","IR",
                            "IRDS","ISA","ISDN","ISH","ISMN","ISQ","IT","ITP","ITU","IUCD","IU","IUD","IUFD","IUGR","IUI","IUP","IUS","IV","IVC","IV-DSA","IVDU",
                            "IVF","IVP","IVPB","IVU","IVUS","JCV","JEV","JIA","JMS","JODM","JRA","JVD","JVP","K","KA","Kcal","KCCT","kg","KIV","KLS","KS","KSHV",
                            "KUB","KVO","L","LA","Lab","LABBB","LAD","LAE","LAH","LAHB","Lam","LAP","LAR","LARP","LAS","Lat","lb; LB","LBBB","LBO","LBP","LBW",
                            "LCA","LCIS","LCM","LCMV","LCV","LCX","L&D","LDH","LDL","LDL-C","L-DOPA","LEC","LEEP","LES","LE","leu","LFT","LGA","LGL","LGM","LGSIL",
                            "LGV","LH","Lig","LIH","LLE","LLETZ","LLL","LLQ","LM","LMA","LMCA","LMD","LMP","LN","LOA","LOC","LOL","LOP","LORTA","LOS","Lot","Lp",
                            "LPH","LPL","LR","LRTI","LT","LTAC","LSB","LSIL","LUL","LUQ","LV","LVAD","LVEDP","LVEF","LVF","LVH","LVOT","Lc of ch","Ly","lytes","M",
                            "MAE","MAHA","MAL","MALT","MAO-I","Mφ","MAP","MARSA","MAS","MAT","MC","MCP","MCHC","MCH","MC&S","MCV","MDD","MDE","MDS","M/E","MEDLINE",
                            "MEN","MeSH","MET","Mg","MgSO4","MGUS","MI","MIC","MICA","MICU","MLC","MLE","MM","M&M","MMPI","mod","MODY","Mo","MOM","MOPP","MPD(s)",
                            "MPV","MR","MRA","MRCP","MRG","MRI","MRSA","MS","MSE","MSH","MSO4","MSU","MSUD","MT","MTBI","MTP","MTX","MVA","MVC","MVP","MVPS","MVR",
                            "Na","NABS","NAD","NAS","NB","NBN","NBTE","NC","NCC","NCS","NCT","NCV","ND","NE","NEC","Neg","Neo","NES","NFR","NGT","NG tube","NGU",
                            "NHL","NICU","NIDDM","NK","NK cells","NKA","NKDA","Nl","NLP","NM","NMR","NNH","NNT","NO","No.","NOF","Non rep.","NOS","NPH","Npl","NPO",
                            "NPTAC","NRB","NRBC","NREM","n.s.","NS","NSA","NSAID","NSCC","NSCLC","NSD","NSE","NSR","NST","NSTEMI","NSU","NT","NTG","NTT","N&V",
                            "n/v","NVD","NVDC","o","O2","OA","OB-GYN","Obl","OBS","Occ","OCD","OCG","OCNA","OCP","OCT","OD","OE","O/E","OGTT","OHL","Oint","OM",
                            "OME","on","OOB","OP","O&P","OPD","OPPT","OPV","OR","ORIF","ORSA","ORT","OS","OSA","OSH","Osm","Osteo","OT","OTC","OTD","OTPP","OU",
                            "oz","p","PA","P&A","PAC","PAD","PAF","PAH","PAI-1","PAL","PALS","PAN","PAO","PAOD","PAP","PARA I","PARA II","PAT","PBF","p.c.",
                            "PCA","PCD","PCI","PCL","PCN","PCNSL","PCO","PCOS","PCP","PCR","PCS","PCV","PCWP","PD","PDA","PDE","PDGF","PDR","PDT","PE","PEA",
                            "PEEP","PEF","PEFR","PEG","pen","PEP","PERRL","PERLA","PERRLA","Per Vag","PET","PFO","PFT","PGCS","PH","PHx","PHTLS","PI","PICC","PID",
                            "PIH","PIP","PKA","PKD","PKU","PLAT","PLT","PM","PMB","PMH","PMI","PML","PMP","PMN","PMR","PM&R","PND","PNH","PNM","PO","POD","poly",
                            "Post","POX","PP","PPCS","PPD","PPH","PPI","PPROM","PPS","Ppt","PPTCT","PPTL","PR","p.r.","PRA","PRBC","Preme","Prep","PRIND","prn",
                            "Prog","PROM","PRP","PRV","PSA","PSH","PSP","PSS","PT","Pt.","PTA","PTB","PTC","PTCA","PTD","PTH","PTHC","PTSD","PTSS","PTT","PTx"
                            ,"PUD","PUO","PUVA","p.v.","PVC (VPC)","PVD","PVFS","PVR","PVS","PWP","Px","P-Y","Q","q.a.d.","QALY","q.AM","q.d.","q.d.s.","q.h.","q.h.s.",
                            "q.i.d.","q.l.","q.m.t.","q.n.","QNS","q.n.s.","q.o.d.","QOF","q.o.h.","q.s.","qt","q.v.","q.w.k.","RA","RAD","Rad hys","RAI","RAPD",
                            "RAS","RBBB","RBC","RBLM","RCA","RCM","RCT","RD","RDS","RDW","REM","RF","RFLP","RFT","r/g/m","Rh","RHD","RhF","RIA","RIBA","RICE","RIMA",
                            "RIND","RL","RLE","RLL","RLN","RLQ","RML","RNA","RNP","RNV","ROM","R/O","ROA","ROP","ROS","ROSC","RPR","RR","RRR","RS cell","RSI","RSV",
                            "RT","RT-PCR","RTA","RTC","RTS","RUE","RUL","RUQ","RV","RVAD","RVF","RVH","RVSP","RVT","Rx or ℞ or Rx","s","Sx","S1","S2","SA","SAB","SAH",
                            "SAN","SAPS II","SAPS III","SARS","SB","SBE","SBO","SBP","s.c.","SCC","SCD","SCLC","SCID","Scope","s.d.","SD","σ","SDH","Sed","Segs","SEM",
                            "SERT","SFA","SGA","SG cath","SGOT","SGPT","SH","SHx","SHBG","SI","SIADH","SICS","SIDS","SIMV","SIT","SK","sl","SLE","SLEV","SLR","SLL",
                            "SM","SMA","SMA-6","SMA-7","SMS","SMT","SMV","SN","SNB","SNP","SNRI","SNV","SOAP","SOB","SOBOE","SOL","SOOB","SOS","SP","s/p","Spec",
                            "SPECT","SPEP","SPET","Sp. fl.","Sp. gr.","SQ","SR","SROM","SS","SSPE","ssRNA","SSRI","SSSS","SSS","ST","Staph.","STD","STAT","STEC",
                            "STEMI","STH","STI","STNR","STOP","Strep.","Strepto.","STS","Subq","Supp","SV","SVC","SVD","SVI","SVN","SVR","SVT","SXA","SXR","Sz",
                            "T","Tx","T&A","T&C","Tab","TAH","TAH-BSO","TB","TBC","TBI","TBLC","TC","TCC","TCM","TCN","TCT","Td","TdP","t.d.s.","TEB","TEE","TEM",
                            "Temp","TENS","TERN","TF","T/F","TFTs","TGA","TG","TGF","TGV","T&H","THR","TIA","TIBC","Tib-Fib","t.i.d.","TIPS","TKO","TKR","TKVO",
                            "TLC","TLR","TM","TMB","TME","TNF","TMJ","TNG","TNM","TOA","TOD","TOE","TOF","TOP","TP","TPa","TPN","TPR","TR","TRAM","TRAP","TRF","TRF'd",
                            "TRH","TS","Tsp","TSH","T.S.T.H.","TT","TTE","TTO","TTP","TTR","TTS","TTTS","Tu","TUR","TURBT","TURP","TVH","UA","UBT","UC","UCHD","UD",
                            "UDS","UE","U&E","UIP","UGI","UOP","Ung","Unk","UPJ","URA","URI","URTI","US","USG","USP","USR","USS","UTI","UVAL","V","VA","VAD","Vag",
                            "VAMP","VBAC","VC","VCTC","vCJD","VD","VDRF","VDRL","VE","VEE","VEB","VF","V-fib","VIP","VLDL","VMA","VNPI","VO","VOD","VPA","VPAP","VPB",
                            "VPC (PVC)","V/Q","VRE","VRSA","VS","VSD","VSR","VT","VTE","VTEC","VZV","w/","WAP","WAT","WBC","WC","W/C","WD","WDL","WH","WDWN","WEE",
                            "WG","WN","WNL","W/O","WPW","WS","wt","WWI","x","XR","XRT","y","YO","ZD","ZIFT"]
        # Set up the PhraseMatcher – it can now take Doc objects as patterns,
        # so even if the list of abbreviations is long, it's very efficient
        patterns = [nlp(org) for org in self.abbreviations]
        self.matcher = PhraseMatcher(nlp.vocab,attr='LOWER')
        self.matcher.add("ABBREVIATION", None, *patterns)

        # Register attribute on the Token. We'll be overwriting this based on
        # the matches, so we're only setting a default value, not a getter.
        Token.set_extension("is_abbreviation", default=True,force=True)

        # Register attributes on Doc and Span via a getter that checks if one of
        # the contained tokens is set to is_abbreviation == True.
        Doc.set_extension("has_abbreviation", getter=self.has_abbreviation)
        Span.set_extension("has_abbreviation", getter=self.has_abbreviation)

    def __call__(self, doc):
        matches = self.matcher(doc)
        #spans = []  # keep the spans for later so we can merge them afterwards        
        seen_tokens=set()
        entities = doc.ents
        new_entities = []
        for _, start, end in matches:
        #    span = Span(doc, start, end, label=match_id)
        #    doc.ents = list(doc.ents) + [span]
            # check for end - 1 here because boundaries are inclusive
            if start not in seen_tokens and end - 1 not in seen_tokens:
                entity =Span(doc, start, end, label=self.label)
                for token in entity:
                    token._.set("is_abbreviation", True)
                new_entities.append(entity)
                entities = [
                    e for e in entities if not (e.start < end and e.end > start)
                ]
                seen_tokens.update(range(start, end))

        doc.ents = tuple(entities) + tuple(new_entities)
        return doc  # don't forget to return the Doc!

    def has_abbreviation(self, tokens):
        return any([t._.get("is_abbreviation") for t in tokens])


class SciSpacyCleansing:
    def __init__(self, apply_rules=False,delimiter='|'):
        self.apply_rules = apply_rules
        self.delimiter = delimiter

        self.__create__()
    
    def __create__(self):
        self.nlp = spacy.load('en_core_sci_lg')
        if self.apply_rules == True:
            self.nlp.add_pipe('custom_abbreviation_detector') 
            ruler = EntityRuler(self.nlp, overwrite_ents=True,name='custom_rules')
            ruler= self.nlp.add_pipe("entity_ruler", config = {'overwrite_ents': True})
            ruler.add_patterns([{"label":"ENTITY","pattern":[{'POS':'ADJ'},{'POS':'NOUN'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"ENTITY","pattern":[{'POS':'ADJ'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'DEP':'neg'},{'POS':'ADJ'},{'POS':'NOUN'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'DEP':'neg'},{'POS':'ADJ'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'DEP':'neg'},{'POS':'NOUN'},{'POS':'NOUN'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'DEP':'neg'},{'POS':'NOUN'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'LOWER':'no','POS':'DET'},{'POS':'ADJ'},{'POS':'NOUN'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'LOWER':'no','POS':'DET'},{'POS':'NOUN'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'LOWER':'no','POS':'DET'},{'POS':'NOUN'}]}])
            ruler.add_patterns([{"label":"NEGATION","pattern":[{'LOWER':'no','POS':'DET'},{'POS':'ADJ'},{'POS':'NOUN'}]}])
            gender = [{'LEMMA':{'IN':['girl','boy','man','woman','lady','women','guy','female','male']}}]
            ruler.add_patterns([{'label':'GENDER','pattern':gender}])
            ruler.add_patterns([{'label':'PERIOD','pattern':[{'POS':'NUM'},{'LEMMA':{'IN':['year','month']}}]}])
            self.nlp.add_pipe('custom_rules',ruler)

            abbreviation = AbbreviationRecognizer(self.nlp)
            self.nlp.add_pipe(abbreviation,first=True)
    
    def __extract_features__(self,doc):
        return self.delimiter.join([ent.text for ent in doc.ents])

    def __process__(self, data):
        result = []
        cores= multiprocessing.cpu_count()-2
        for doc in tqdm(self.nlp.pipe(data,batch_size=500, n_process= cores), total = len(data)):
            #result.append(self.__extract_features__(doc))
            result.append(self.delimiter.join([ent.text for ent in doc.ents]))
        return result

    def run(self,data_df,column_name,processed_column_name):        
        data_df[processed_column_name] = self.__process__(data_df[column_name].tolist())
        return data_df

    