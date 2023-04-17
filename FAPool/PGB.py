import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
import encoding
from encoding.functions import scaled_l2, aggregate, pairwise_cosine


class PGB(Module):
    def __init__(self, D, K):
        super(PGB, self).__init__()
        # init codewords and smoothing factor
        self.D, self.K = D, K
        self.codewords = Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std1 = 1./((self.K*self.D)**(1/2))
        self.codewords.data.uniform_(-std1, std1)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):
        # input X is a 4D tensor
        assert(X.size(1) == self.D)
        B, D = X.size(0), self.D
        H, W = X.size(2), X.size(3)
        if X.dim() == 3:
            # BxDxN => BxNxD
            X = X.transpose(1, 2).contiguous()
        elif X.dim() == 4:
            # BxDxHxW => Bx(HW)xD
            X = X.view(B, D, -1).transpose(1, 2).contiguous()
        else:
            raise RuntimeError('Encoding Layer unknown input dims!')
        # assignment weights BxNxK
        # import pdb
        # pdb.set_trace()
        #A = F.softmax(X, dim=2).expand(1,225,16)
        A = F.softmax(scaled_l2(X, self.codewords, self.scale), dim=2)
       # print("sadafsadfagfagag")
        
        #print(A.shape)
        # aggregate
        # E = aggregate(A, X, self.codewords)
        A = A.permute(0,2,1).contiguous()
        A = A.view(B,self.K,H,W)
        return A

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.D) + '=>' + str(self.K) + 'x' \
            + str(self.D) + ')'
