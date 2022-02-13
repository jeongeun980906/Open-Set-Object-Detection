import torch

def mln_gather(batch_out):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    pi = batch_out['pi']
    mu = batch_out['mu']

    max_idx = torch.argmax(pi,dim=1) # [N]
    mu      = torch.softmax(mu,dim=2) #[N x K x D]
    idx_gather = max_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel = torch.gather(mu,dim=1,index=idx_gather).squeeze(dim=1) # [N x D]
    # sigma_sel = torch.gather(sigma,dim=1,index=idx_gather).squeeze(dim=1) # [N x D]
    # out = {'max_idx':max_idx, # [N]
    #        'idx_gather':idx_gather, # [N x 1 x D]
    #        'mu_sel':mu_sel, # [N x D]
    #        'sigma_sel':sigma_sel # [N x D]
    #        }
    return mu_sel

def mace_loss(scores, gt_classes, reduction="mean"):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
        :param target:  [N x D]
    """
    mu = scores['mu']
    pi = scores['pi']
    sigma = scores['sigma']
    # $\mu$
    mu_hat = torch.softmax(mu,dim=2) # logit to prob [N x K x D]
    log_mu_hat = torch.log(mu_hat+1e-6) # [N x K x D]
    # $\pi$
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(mu) # [N x K x D]
    # target
    target_usq =  torch.unsqueeze(gt_classes,1) # [N x 1 x D]
    target_exp =  target_usq.expand_as(mu) # [N x K x D]
    # CE loss
    ce_exp = -target_exp*log_mu_hat # CE [N x K x D]
    ace_exp = ce_exp / sigma # attenuated CE [N x K x D]
    mace_exp = torch.mul(pi_exp,ace_exp) # mixtured attenuated CE [N x K x D]
    ce_exp = torch.mul(pi_exp,ce_exp)
    ce=torch.sum(ce_exp,dim=1) # [N x D]
    ce=torch.sum(ce,dim=1) # [N]
    mace = torch.sum(mace_exp,dim=1) # [N x D]
    mace = torch.sum(mace,dim=1) # [N]
    mace_avg = torch.mean(mace) # [1]
    ce_avg=torch.mean(ce) # [1]
    # Compute uncertainties (epis and alea)
    unct_out = mln_uncertainties(pi,mu,sigma)
    epis = unct_out['epis'] # [N]
    alea = unct_out['alea'] # [N]
    pi_entropy = unct_out['pi_entropy'] # [N]
    epis_avg = torch.mean(epis) # [1]
    alea_avg = torch.mean(alea) # [1]
    # pi_entropy_avg=torch.mean(pi_entropy) # [1]
    # Return
    if reduction == 'mean':
        return mace_avg - epis_avg + alea_avg
    elif reduction == 'sum':
        return torch.sum(mace - epis + alea)
    else:
        raise NotImplementedError

def mln_uncertainties(pi,mu,sigma):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    # entropy of pi
    entropy_pi  = -pi*torch.log(pi+1e-8)
    entropy_pi  = torch.sum(entropy_pi,1) #[N]
    # $\pi$
    mu_hat = torch.softmax(mu,dim=2) # logit to prob [N x K x D]
    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]
    # softmax($\mu$) average
    mu_hat_avg = torch.sum(torch.mul(pi_exp,mu_hat),dim=1).unsqueeze(1) # [N x 1 x D]
    mu_hat_avg_exp = mu_hat_avg.expand_as(mu) # [N x K x D]
    mu_hat_diff_sq = torch.square(mu_hat-mu_hat_avg_exp) # [N x K x D]
    # Epistemic uncertainty
    epis = torch.sum(torch.mul(pi_exp,mu_hat_diff_sq), dim=1)  # [N x D]
    epis = torch.sqrt(torch.sum(epis,dim=1)+1e-6) # [N]
    # Aleatoric uncertainty
    alea = torch.sum(torch.mul(pi_exp,sigma), dim=1)  # [N x D]
    alea = torch.sqrt(torch.mean(alea,dim=1)+1e-6) # [N]
    # Return
    unct_out = {'epis':epis, # [N]
                'alea':alea,  # [N]
                'pi_entropy':entropy_pi
                }
    return unct_out