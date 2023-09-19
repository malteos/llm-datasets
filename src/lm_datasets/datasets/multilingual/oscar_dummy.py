from lm_datasets.datasets.base import BaseDataset, Availability

# From https://docs.google.com/spreadsheets/d/1_rfLKa_Kq09YI0BPnfmoSL6-U3SHrW8tRwmk3-Qchzo/edit#gid=430577929
OSCAR_RAW_LANG_TO_TOKENS = """af	6,217,024
am	40,262,809
an	264
ar	10,081,452,882
arz	385,511
as	24,395,215
ast	10,917
av	1,073
az	316,850,330
azb	2,029,729
ba	26,036,637
be	164,729,607
bg	3,635,273,738
bh	507
bn	1,092,983,765
bo	6,062,558
bpy	346,947
br	4,759,407
bs	395
bxr	698
ca	2,240,460,836
ce	1,051,752
ceb	6,263,404
ckb	61,334,746
cs	9,717,378,559
cv	3,039,925
cy	52,250,043
da	2,217,634,340
de	73,848,586,648
dsb	84
dv	10,655,359
el	7,691,622,692
en	523,869,288,690
eo	67,774,923
es	63,388,237,965
et	938,296,892
eu	136,672,615
fa	9,609,206,698
fi	4,198,143,883
fr	62,127,088,294
fy	9,885,788
ga	9,054,923
gd	18,458
gl	38,345,625
gn	260
gom	121,035
gsw	34,328
gu	417,001,705
he	1,697,158,891
hi	2,475,605,444
hr	3,542,961
hsb	15,827
ht	20,671
hu	16,013,364,289
hy	336,045,041
ia	9,384
id	3,228,020,221
ie	0
ilo	4,411
io	2,598
is	294,471,539
it	36,327,274,203
ja	4,401,059,165
jbo	312,315
jv	3,286
ka	373,935,158
kk	214,679,857
km	59,880,231
kn	124,924,350
ko	3,435,866,935
krc	8,385
ku	25,921,607
kv	5,909
kw	80
ky	32,062,783
la	307,865
lb	2,514,838
lez	60,634
li	169
lmo	3,269
lo	10,659,203
lt	1,674,362,574
lv	845,459,899
mai	467
mg	1,416,430
mhr	1,615,215
min	305,589
mk	389,344,425
ml	236,597,838
mn	454,350,415
mr	252,706,331
mrj	60,180
ms	238,477
mt	149,886
multi	1,251,676,406
mwl	54
my	82,433,836
mzn	16,115
nah	279
nds	1,612,175
ne	278,901,036
new	229,703
nl	19,564,553,306
nn	575,518
no	373,160,033
oc	34,701
or	31,963,340
os	3,935,964
pa	104,235,418
pl	18,073,705,588
pms	510,087
pnb	11,844,695
ps	30,196,179
pt	15,172,557,311
qu	13
ro	6,302,600,833
ru	78,032,029,344
sa	2,479,345
sah	4,288,051
sd	14,667,207
sh	166,517
si	172,755,385
sk	2,704,716,280
sl	192,816,743
so	51
sq	462,694,599
sr	632,781,822
su	258
sv	6,993,719,601
sw	164,459
ta	738,824,392
te	201,575,815
tg	76,987,285
th	2,224,483,018
tk	325,786
tl	110,560,444
tr	8,290,890,087
tt	59,253,765
ug	14,659,554
uk	3,183,842,018
ur	434,023,273
uz	1,665,960
vi	22,424,984,210
vo	49,968
wa	6,347
war	19,665
wuu	1,199
x-eml	329
xal	27
xmf	283,316
yi	14,287,370
yo	2,396
zh	44,378,380,161"""

RAW_FILTER_LANGUAGES = """bg
cs
da
de
el
en
es
et
fi
fr
ga
hr
hu
it
lt
lv
mt
nl
pl
pt
ro
sk
sl
sv
code
uk
sr
sh
nn
no"""


class OscarBaseDataset(BaseDataset):
    SOURCE_ID = "oscar"
    DESCRIPTION = (
        "The OSCAR project (Open Super-large Crawled Aggregated coRpus) is an Open Source project aiming to provide"
        " web-based multilingual resources and datasets for Machine Learning (ML) and Artificial Intelligence (AI)"
        " applications."
    )
    HOMEPAGE = "https://oscar-project.org/"
    AVAILIBILITY = Availability.ON_REQUEST
    WEB_CRAWLED = True
    DUMMY = True


def get_oscar_dummy_cls_by_language(lang, tokens):
    class OscarLanguageDataset(OscarBaseDataset):
        TOKENS = tokens
        DATASET_ID = "oscar_" + lang
        TITLE = "oscar_" + lang
        LANGUAGES = [lang]

    return OscarLanguageDataset


def get_oscar_dummy_classes():
    """
    Generate dummy dataset classes with token count based on OSCAR 23.01
    """
    pass

    lang_to_tokens = {
        row.split("\t")[0]: int(row.split("\t")[1].replace(",", "")) for row in OSCAR_RAW_LANG_TO_TOKENS.splitlines()
    }

    filter_langs = set(RAW_FILTER_LANGUAGES.splitlines())

    return [
        get_oscar_dummy_cls_by_language(lang, tokens * 10)  # TODO token count like OSCAR based on ten CC dumps
        for lang, tokens in lang_to_tokens.items()
        if lang in filter_langs
    ]
