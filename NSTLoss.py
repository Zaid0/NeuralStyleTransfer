import torch.nn as nn
import torch

class NSTLoss(nn.Module):
    def __init__(self, alpha, beta, wl, return_gram=False):
        super(NSTLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.wl = wl
        self.return_gram = return_gram

    def content_loss(self, content_features, generated_features):
        content_loss = torch.mean((content_features - generated_features) ** 2)
        return  content_loss

    def gram_matrix(self, imgfeature):
        _, d, h, w = imgfeature.size()
        imgfeature = imgfeature.view(d, h * w)
        gram_mat = torch.mm(imgfeature, imgfeature.t())
        return gram_mat

    def style_loss(self, style_features, generated_features):
        A = self.gram_matrix(style_features)
        if self.return_gram:
            return A
        G = self.gram_matrix(generated_features)
        _, channels, hieght, width = generated_features.shape
        E = torch.mean((G - A)**2) / channels*hieght*width
        return E

    def forward(self, content_features, style_features, generated_content_features, generated_style_features):
        Gram_matrix = []
        if self.return_gram:
            for sty_features in style_features:
                Gram_matrix.append(
                    self.style_loss(sty_features,
                                     None)
                )
            return Gram_matrix

        C_Loss = self.content_loss(content_features, generated_content_features)
        S_Loss = 0
        j = 0
        for sty_features, gen_features in zip(
                style_features, generated_style_features
        ):
            S_Loss += (
                    self.wl[j] * (self.style_loss(sty_features,gen_features))
            )
            j+=1

        loss = self.alpha*C_Loss + self.beta*S_Loss
        return loss
