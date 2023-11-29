import torch
def kl_for_log_probs(log_p, log_q):
  p = torch.exp(log_p)
  neg_ent = torch.sum(p * log_p, dim=-1)
  neg_cross_ent = torch.sum(p * log_q, dim=-1)
  kl = neg_ent - neg_cross_ent
  return kl

p = torch.tensor([0.4,0.6])
q = torch.tensor([0.3,0.7])


def dkl(_p,_q):
    return torch.sum(_p*(_p.log()-_q.log()),dim=-1)

print(kl_for_log_probs(p.log(), q.log()))

print(dkl(p,q))
print(torch.nn.functional.kl_div(q, p))
print(torch.nn.functional.kl_div(q, p,reduction='sum'))
print(torch.nn.functional.kl_div(q.log(), p,reduction='sum'))
print(torch.nn.functional.kl_div(q.softmax(-1).log(), p.softmax(-1),reduction='sum'))

# print(torch.nn.functional.kl_div(p, q))
# print(torch.nn.functional.kl_div(p, q,reduction='sum'))
# print(torch.nn.functional.kl_div(p.log(), q,reduction='sum'))
# print(torch.nn.functional.kl_div(p.softmax(-1).log(), q.softmax(-1),reduction='sum'))