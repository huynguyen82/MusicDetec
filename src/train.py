import torch
torch.manual_seed(123)
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from config import RANKS, DATAPATH, MODELPATH, MODELCHECKPOINT
from model import genreNet
from data import Data_rank
from set import Set
import os
import re
import math

def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

def main():
    # ------------------------------------------------------------------------------------------- #
    ## DATA
    #data    = Data_rank(RANKS, DATAPATH)
    #data.make_raw_data()    # computing features
    #data.make_data_sets()   
    
    # ------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------- #
    ## SET
    data    = Data_rank(RANKS, DATAPATH)
    set_                = Set(data)    
    set_.chunk_size     = 40 # files
    set_.make_chunks()   

    #x_train, y_train    = set_.get_train_set()
    #TRAIN_SIZE  = len(x_train)
    #     # ------------------------------------------------------------------------------------------- #   
    
      
    #rates = [5e-1,5e-2,5e-3,5e-4,5e-5]

    EPOCH_NUM   = 250
    BATCH_SIZE  = 64
    ##
     # for active learning rate
    lr_max = 3*10e-3
    factor = 6
    end_lr = lr_max        
    len_train = 0
    if os.path.isfile('length'):
        len_train = torch.load('length')
    else:
        print('Getting length of training data')
        _count = 0
        _countT = len(set_.train_chunks)
        for chunk in set_.train_chunks:
            _count +=1        
            x_train, y_train    = set_.get_set(chunk)
            len_train += len(x_train)
            print('Done for get length of training data for the chunk: ' + str(_count) + '/' + str(_countT))
        torch.save(len_train,'length')
    len_train=int(len_train/BATCH_SIZE)
    step_size = 4*len_train
    clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
    # load check point if existing    
    point = 0                                    
    ##
    net = genreNet()
    optimizer   = torch.optim.RMSprop(net.parameters(),momentum=0.9)
    #optimizer   = torch.optim.Adam(net.parameters(),lr=1e-2, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=[EPOCH_NUM,0.95])    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)    
    if not os.path.isfile(MODELCHECKPOINT):
        print("Starting from zero point")                         
        point = -1
    else:
        checkpoint = torch.load(MODELCHECKPOINT)
        net = checkpoint['model']
        optimizer   = torch.optim.Adam(net.parameters())
        point = checkpoint['epoch']
        net.load_state_dict(checkpoint['state_dict'])                
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Starting from the checkpoint " + MODELCHECKPOINT + ' from epoch: ' + str(point))        
    
    criterion   = torch.nn.CrossEntropyLoss()   
    net.cuda()       
    #
    for epoch in range(EPOCH_NUM):
        if epoch > point:  
            c_iter = 0;   
            num_chunks = len(set_.train_chunks)            
            for chunk in set_.train_chunks:                
                x_train, y_train    = set_.get_set(chunk)                
                if x_train is not None:
                    TRAIN_SIZE  = len(x_train)
                    inp_, out_    = Variable(torch.from_numpy(x_train)).float().cuda(), Variable(torch.from_numpy(y_train)).long().cuda()                    
                    #inp_valid, out_valid    = Variable(torch.from_numpy(x_valid)).float().cuda(), Variable(torch.from_numpy(y_valid)).long().cuda()
                    # ------------------------------------------------------------------------------------------------- #
                    ## TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE # TRAIN PHASE  #
                    # ------------------------------------------------------------------------------------------------- #
                    train_loss = 0                   
                    for i in range(0, TRAIN_SIZE, BATCH_SIZE):                        
                        x_train_batch, y_train_batch = inp_[i:i + BATCH_SIZE], out_[i:i + BATCH_SIZE]
                        pred_train_batch    = net(x_train_batch)
                        loss_train_batch    = criterion(pred_train_batch, y_train_batch)                          
                        train_loss          += loss_train_batch.data.cpu().item()
                        loss_train_batch.backward()                                        
                    #clip_grad_norm_(net.parameters(), 5)                                    
                    optimizer.step()  # <-- OPTIMIZER                                  
                    optimizer.zero_grad()  # <-- OPTIMIZER

                    epoch_train_loss    = (train_loss * BATCH_SIZE) / TRAIN_SIZE
                    train_sum           = 0
                    for i in range(0, TRAIN_SIZE, BATCH_SIZE):
                        pred_train      = net(inp_[i:i + BATCH_SIZE])
                        indices_train   = pred_train.max(1)[1]
                        train_sum       += (indices_train == out_[i:i + BATCH_SIZE]).sum().data.cpu().item()
                    train_accuracy  = train_sum / float(TRAIN_SIZE)
                    print("Epoch: %d\t\tIter: %d/%d\t\tTrain loss : %.2f\t\tTrain accuracy: %.2f" % \
                        (epoch + 1, c_iter+1, num_chunks, epoch_train_loss, train_accuracy))
                    c_iter+=1
                    #                    
                    del inp_, out_
            scheduler.step()
            
            # ------------------------------------------------------------------------------------------------- #
            ## SAVE checkpoint
            # ------------------------------------------------------------------------------------------------- #
            checkpoint = {'model': genreNet(),
                        'state_dict': net.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'epoch' : epoch}
            torch.save(checkpoint, MODELCHECKPOINT)
            print("Saving checkpoint for epoch " + str(epoch))
            # ------------------------------------------------------------------------------------------------- #
            # ------------------------------------------------------------------------------------------------- #
            ## VALIDATION PHASE ## VALIDATION PHASE ## VALIDATION PHASE ## VALIDATION PHASE ## VALIDATION PHASE #
            # ------------------------------------------------------------------------------------------------- #
            torch.cuda.empty_cache()
            valid_loss = 0
            valid_sum  = 0            
            for v_chunk in set_.valid_chunks:
                x_valid, y_valid = set_.get_set(v_chunk)                  
                if x_valid is not None:              
                    VALID_SIZE  = len(x_valid)
                    inp_, out_    = Variable(torch.from_numpy(x_valid)).float().cuda(), Variable(torch.from_numpy(y_valid)).long().cuda()                    
                    for i in range(0, VALID_SIZE, BATCH_SIZE):
                        x_valid_batch, y_valid_batch = inp_[i:i + BATCH_SIZE], out_[i:i + BATCH_SIZE]
                        pred_valid_batch    = net(x_valid_batch)
                        loss_valid_batch    = criterion(pred_valid_batch, y_valid_batch)
                        valid_loss          += loss_valid_batch.data.cpu().item()
                
                    for i in range(0, VALID_SIZE, BATCH_SIZE):
                        pred_valid      = net(inp_[i:i + BATCH_SIZE])
                        indices_valid   = pred_valid.max(1)[1]
                        valid_sum       += (indices_valid == out_[i:i + BATCH_SIZE]).sum().data.cpu().item()                    
                    del inp_, out_
            valid_accuracy  = valid_sum / float(VALID_SIZE)
            epoch_valid_loss    = (valid_loss * BATCH_SIZE) / VALID_SIZE
            print("Epoch: %d\t\tTrain loss : %.2f\t\tValid loss : %.2f\t\tTrain acc : %.2f\t\tValid acc : %.2f" % \
                (epoch + 1, epoch_train_loss, epoch_valid_loss, train_accuracy, valid_accuracy))
            # ------------------------------------------------------------------------------------------------- #
            
    
    torch.save(net.state_dict(), MODELPATH)
    print('-> ptorch model is saved.')
    # ------------------------------------------------------------------------------------------------- #
    ## EVALUATE TEST ACCURACY
    # ------------------------------------------------------------------------------------------------- #
    torch.cuda.empty_cache()
    test_sum = 0
    for t_chunk in set_.test_chunks:
        x_test, y_test = set_.get_set(t_chunk)
        if x_test is not None:
            TEST_SIZE   = len(x_test)
            inp_, out_ = Variable(torch.from_numpy(x_test)).float().cuda(), Variable(torch.from_numpy(y_test)).long().cuda()    
            for i in range(0, TEST_SIZE, BATCH_SIZE):
                pred_test       = net(inp_[i:i + BATCH_SIZE])
                indices_test    = pred_test.max(1)[1]
                test_sum        += (indices_test == out_[i:i + BATCH_SIZE]).sum().data.cpu().item()            
            del inp_, out_
    test_accuracy   = test_sum / float(TEST_SIZE)
    print("Test acc: %.2f" % test_accuracy)
    # ------------------------------------------------------------------------------------------------- #

    return

if __name__ == '__main__':
    main()





