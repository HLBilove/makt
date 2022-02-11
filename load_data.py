# Code reused from https://github.com/arghosh/AKT

import numpy as np
import math
class DATA(object):
    def __init__(self, n_skill, seqlen, separate_char, name="data"):
        self.separate_char = separate_char
        self.seqlen = seqlen
        self.n_skill = n_skill

    def load_data(self, path):
        file_data = open(path, 'r')
        s_data = []
        sr_data = []
        sa1_data = []
        sa2_data = []
        e_data = []
        if_data = []
        for lineID, line in enumerate(file_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 6 == 0:
                learner_id = lineID//3
            if lineID % 6 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
            if lineID % 6 == 2:
                E = line.split(self.separate_char)
                if len(E[len(E)-1]) == 0:
                    E = E[:-1]
            if lineID % 6 == 3:
                IF = line.split(self.separate_char)
                if len(IF[len(IF)-1]) == 0:
                    IF = IF[:-1]
            if lineID % 6 == 4:
                R = line.split(self.separate_char)
                if len(R[len(R)-1]) == 0:
                    R = R[:-1]
            if lineID % 6 == 5:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                A1 = []
                A2 = []
                for i in range(len(A)):
                    A1.append(A[i].split(' ')[0])
                    A2.append(A[i].split(' ')[1])
                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    s_seq = []
                    e_seq = []
                    if_seq = []
                    r_seq = []
                    a1_seq = []
                    a2_seq = []
                    if k == n_split - 1:
                        endINdex = len(R)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            Xindex = int(S[i]) + round(float(R[i])) * self.n_skill
                            Yindex = int(S[i]) + round(float(A1[i])) * self.n_skill
                            Zindex = int(S[i]) + round(float(A2[i])) * self.n_skill
                            s_seq.append(int(S[i]))
                            e_seq.append(int(E[i]))
                            if_seq.append(int(IF[i]))
                            r_seq.append(Xindex)
                            a1_seq.append(Yindex)
                            a2_seq.append(Zindex)
                        else:
                            print(S[i])
                    s_data.append(s_seq)
                    sr_data.append(r_seq)
                    sa1_data.append(a1_seq)
                    sa2_data.append(a2_seq)
                    e_data.append(e_seq)
                    if_data.append(if_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        s_dataArray = np.zeros((len(s_data), self.seqlen))
        for j in range(len(s_data)):
            dat = s_data[j]
            s_dataArray[j, :len(dat)] = dat

        sr_dataArray = np.zeros((len(sr_data), self.seqlen))
        for j in range(len(sr_data)):
            dat = sr_data[j]
            sr_dataArray[j, :len(dat)] = dat

        sa1_dataArray = np.zeros((len(sa1_data), self.seqlen))
        for j in range(len(sa1_data)):
            dat = sa1_data[j]
            sa1_dataArray[j, :len(dat)] = dat

        sa2_dataArray = np.zeros((len(sa2_data), self.seqlen))
        for j in range(len(sa2_data)):
            dat = sa2_data[j]
            sa2_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        if_dataArray = np.zeros((len(if_data), self.seqlen))
        for j in range(len(if_data)):
            dat = if_data[j]
            if_dataArray[j, :len(dat)] = dat

        return s_dataArray, sr_dataArray, sa1_dataArray, sa2_dataArray, e_dataArray, if_dataArray

    def load_test_data(self, path):
        file_data = open(path, 'r')
        s_data = []
        sr_data = []
        sa1_data = []
        sa2_data = []
        e_data = []
        if_data = []
        test_e_num = 0
        for lineID, line in enumerate(file_data):
            line = line.strip()
            # lineID starts from 0
            if lineID % 6 == 0:
                learner_id = lineID//3
            if lineID % 6 == 1:
                S = line.split(self.separate_char)
                if len(S[len(S)-1]) == 0:
                    S = S[:-1]
                test_e_num += len(S)
            if lineID % 6 == 2:
                E = line.split(self.separate_char)
                if len(E[len(E)-1]) == 0:
                    E = E[:-1]
            if lineID % 6 == 3:
                IF = line.split(self.separate_char)
                if len(IF[len(IF)-1]) == 0:
                    IF = IF[:-1]
            if lineID % 6 == 4:
                R = line.split(self.separate_char)
                if len(R[len(R)-1]) == 0:
                    R = R[:-1]
            if lineID % 6 == 5:
                A = line.split(self.separate_char)
                if len(A[len(A)-1]) == 0:
                    A = A[:-1]
                A1 = []
                A2 = []
                for i in range(len(A)):
                    A1.append(A[i].split(' ')[0])
                    A2.append(A[i].split(' ')[1])

                # start split the data
                n_split = 1
                if len(S) > self.seqlen:
                    n_split = math.floor(len(S) / self.seqlen)
                    if len(S) % self.seqlen:
                        n_split = n_split + 1

                for k in range(n_split):
                    s_seq = []
                    e_seq = []
                    if_seq = []
                    r_seq = []
                    a1_seq = []
                    a2_seq = []
                    if k == n_split - 1:
                        endINdex = len(R)
                    else:
                        endINdex = (k+1) * self.seqlen
                    for i in range(k * self.seqlen, endINdex):
                        if len(S[i]) > 0:
                            Xindex = int(S[i]) + round(float(R[i])) * self.n_skill
                            Yindex = int(S[i]) + round(float(A1[i])) * self.n_skill
                            Zindex = int(S[i]) + round(float(A2[i])) * self.n_skill
                            s_seq.append(int(S[i]))
                            e_seq.append(int(E[i]))
                            if_seq.append(int(IF[i]))
                            r_seq.append(Xindex)
                            a1_seq.append(Yindex)
                            a2_seq.append(Zindex)
                        else:
                            print(S[i])
                    s_data.append(s_seq)
                    sr_data.append(r_seq)
                    sa1_data.append(a1_seq)
                    sa2_data.append(a2_seq)
                    e_data.append(e_seq)
                    if_data.append(if_seq)

        file_data.close()
        ### data: [[],[],[],...] <-- set_max_seqlen is used
        # convert data into ndarrays for better speed during training
        s_dataArray = np.zeros((len(s_data), self.seqlen))
        for j in range(len(s_data)):
            dat = s_data[j]
            s_dataArray[j, :len(dat)] = dat

        sr_dataArray = np.zeros((len(sr_data), self.seqlen))
        for j in range(len(sr_data)):
            dat = sr_data[j]
            sr_dataArray[j, :len(dat)] = dat

        sa1_dataArray = np.zeros((len(sa1_data), self.seqlen))
        for j in range(len(sa1_data)):
            dat = sa1_data[j]
            sa1_dataArray[j, :len(dat)] = dat

        sa2_dataArray = np.zeros((len(sa2_data), self.seqlen))
        for j in range(len(sa2_data)):
            dat = sa2_data[j]
            sa2_dataArray[j, :len(dat)] = dat

        e_dataArray = np.zeros((len(e_data), self.seqlen))
        for j in range(len(e_data)):
            dat = e_data[j]
            e_dataArray[j, :len(dat)] = dat

        if_dataArray = np.zeros((len(if_data), self.seqlen))
        for j in range(len(if_data)):
            dat = if_data[j]
            if_dataArray[j, :len(dat)] = dat

        return s_dataArray, sr_dataArray, sa1_dataArray, sa2_dataArray, e_dataArray, if_dataArray, test_e_num