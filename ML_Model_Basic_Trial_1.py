import pandas as pd

df = pd.read_csv('ML_Data_Two_Variables_Only.txt', names=['Title_Abstract', 'Status'], sep='\t')

from sklearn.model_selection import train_test_split
Title_Abstract = df['Title_Abstract'].values
labels = df['Status'].values
Title_Abstract_train, Title_Abstract_test, labels_train, labels_test = train_test_split(Title_Abstract, labels,
                                                                test_size=0.2, random_state=1000)

from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
punctuations = string.punctuation
parser = English()
stopwords = list(STOP_WORDS)


def spacy_tokenizer(utterance):
    tokens = parser(utterance)
    return [token.lemma_.lower().strip() for token in tokens if
            token.text.lower().strip() not in stopwords and token.text not in punctuations]

from sklearn.feature_extraction.text import CountVectorizer
#vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
vectorizer = CountVectorizer()
vectorizer.fit(Title_Abstract_train)

X_train = vectorizer.transform(Title_Abstract_train)
X_test = vectorizer.transform(Title_Abstract_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver="lbfgs", max_iter=1000)
classifier.fit(X_train, labels_train)

accuracy = classifier.score(X_test, labels_test)
print("The accuracy of the prediction of this model is:", accuracy)

Query_Title_1 = "MicroRNA-34 suppresses proliferation of human ovarian cancer cells by triggering autophagy and apoptosis and inhibits cell invasion by targeting Notch 1"
Query_Abstract_1 = "Ovarian cancer is one the prevalent cancers in women and is responsible for 5% of all the cancer related mortalities in women. Owing to late diagnosis, frequent relapses, side effects of chemotherapy, development of drug resistance, there is pressing need to screen out novel and effective treatment options. Accumulating evidences suggest that miRNAs may prove essential therapeutic targets for the treatment of cancer. This study was designed to investigate the role and therapeutic potential of miR-34 in ovarian cancer. It was found that miR-34 is significantly downregulated in ovarian cancer cell lines. Overexpression of miR-34 causes significant decrease in the proliferation of OVACAR-3 ovarian cancer cells via activation of apoptosis and autophagy. The miR-34 overexpression was concomitant with upsurge of apoptosis related proteins (Bax) and the autophagy associated protein (LC3 II and p62). TargetScan analysis showed Notch 1 to be the main target of miR-34 in OVACAR-3 cells which was further validated by luciferase reporter assay. The qRT-PCR results showed Notch 1 to be 3.2-4.1 fold higher in the ovarian cancer cell lines relative to the non-cancerous cells. Nonetheless, miR-34 overexpression in OVACAR-3 cells resulted in the post-transcriptional suppression of Notch 1 expression. Silencing of Notch 1 also caused inhibition of OVACAR-3 cell proliferation via induction of apoptosis and autophagy. Overexpression of Notch 1 could partially rescue the effects of miR-34 overexpression on the proliferation of OVACAR-3 cells. Moreover, overexpression of miR-34 causes significant inhibition of the invasion of the OVACAR-3 cells. The findings of the present study indicate the tumor suppressive role of miR-34 in ovarian cancer and may therefore prove to be a potential therapeutic target."

Query_Title_2 = "Umbilical mesenchymal stem cell-derived exosomes facilitate spinal cord functional recovery through the miR-199a-3p/145-5p-mediated NGF/TrkA signaling pathway in rats"
Query_Abstract_2 = "Background: Although exosomes, as byproducts of human umbilical cord mesenchymal stem cells (hUC-MSCs), have been demonstrated to be an effective therapy for traumatic spinal cord injury (SCI), their mechanism of action remains unclear. Methods: We designed and performed this study to determine whether exosomes attenuate the lesion size of SCI by ameliorating neuronal injury induced by a secondary inflammatory storm and promoting neurite outgrowth. We determined the absolute levels of all exosomal miRNAs and investigated the potential mechanisms of action of miR-199a-3p/145-5p in inducing neurite outgrowth in vivo and in vitro. Results: miR-199a-3p/145-5p, which are relatively highly expressed miRNAs in exosomes, promoted PC12 cell differentiation suppressed by lipopolysaccharide (LPS) in vitro through modulation of the NGF/TrkA pathway. We also demonstrated that Cblb was a direct target of miR-199a-3p and that Cbl was a direct target of miR-145-5p. Cblb and Cbl gene knockdown resulted in significantly decreased TrkA ubiquitination levels, subsequently activating the NGF/TrkA downstream pathways Akt and Erk. Conversely, overexpression of Cblb and Cbl was associated with significantly increased TrkA ubiquitination level, subsequently inactivating the NGF/TrkA downstream pathways Akt and Erk. Western blot and coimmunoprecipitation assays confirmed the direct interaction between TrkA and Cblb and TrkA and Cbl. In an in vivo experiment, exosomal miR-199a-3p/145-5p was found to upregulate TrkA expression at the lesion site and also promote locomotor function in SCI rats. Conclusions: In summary, our study showed that exosomes transferring miR-199a-3p/145-5p into neurons in SCI rats affected TrkA ubiquitination and promoted the NGF/TrkA signaling pathway, indicating that hUC-MSC-derived exosomes may be a promising treatment strategy for SCI."

Query_Title_3 = "rs41291957 controls miR-143 and miR-145 expression and impacts coronary artery disease risk"
Query_Abstract_3 = "The role of single nucleotide polymorphisms (SNPs) in the etiopathogenesis of cardiovascular diseases is well known. The effect of SNPs on disease predisposition has been established not only for protein coding genes but also for genes encoding microRNAs (miRNAs). The miR-143/145 cluster is smooth muscle cell-specific and implicated in the pathogenesis of atherosclerosis. Whether SNPs within the genomic sequence of the miR-143/145 cluster are involved in cardiovascular disease development is not known. We thus searched annotated sequence databases for possible SNPs associated with miR-143/145. We identified one SNP, rs41291957 (G > A), located -91 bp from the mature miR-143 sequence, as the nearest genetic variation to this miRNA cluster, with a minor allele frequency > 10%. In silico and in vitro approaches determined that rs41291957 (A) upregulates miR-143 and miR-145, modulating phenotypic switching of vascular smooth cells towards a differentiated/contractile phenotype. Finally, we analysed association between rs41291957 and CAD in two cohorts of patients, finding that the SNP was a protective factor. In conclusion, our study links a genetic variation to a pathological outcome through involvement of miRNAs."

string_1 = "Title: " + Query_Title_1 + ". Abstract: " + Query_Abstract_1
string_2 = "Title: " + Query_Title_2 + ". Abstract: " + Query_Abstract_2
string_3 = "Title: " + Query_Title_3 + ". Abstract: " + Query_Abstract_3
new_reviews = [string_1, string_2, string_3]
X_new = vectorizer.transform(new_reviews)

for x in classifier.predict(X_new):
    print("This VERY basic model that has a lot to be improved on and has yet to be properly tested predicts this paper has: ")
    if int(x) == 1:
        print("Sequence errors")
    else:
        print("No sequence errors")

#print(classifier.predict(X_new))
