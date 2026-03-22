import torch
p=3
n=8
tn=n+p+1
t=torch.linspace(-1,1,8)
print(len(t))
c=torch.randn(len(t)-p)
order=3
x=0.1


from spline import cox_deboor
print(cox_deboor(c,t,order,x))