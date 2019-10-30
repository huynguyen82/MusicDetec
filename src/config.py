# 10 GENRES USED FOR CLASSIFICATION
GENRES      = [ 'blues',
                'classical',
                'country',
                'disco',
                'hiphop',
                'jazz',
                'metal',
                'pop',
                'reggae',
                'rock'  ]

RANKS      = [ '1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                '10'  ]

# DEFINE PATHS
DATARANK        = '../utils/train_rank'
DATAGEN         = '../utils/train_data'
DATAFEAT        = '../data/'
#DATAGEN         = '../utils/vd'
DATAPATH        =   '../utils/'
RAW_DATAPATH    =   '../utils/raw_data.pkl'
SET_DATAPATH    =   '../utils/set.pkl'
MODELPATH_feat  =   '../utils/net_feat.pt'
MODELPATH       =   '../utils/net.pt'
MODELCHECKPOINT =   '../utils/checkpoint.pth'
DATAJOBS        = 20