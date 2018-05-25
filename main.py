import nimfa
import numpy as np

V = nimfa.examples.medulloblastoma.read(normalize=True)

consensus_matrix = []
cophenetic_coeff = []
for x in range(2, 50):
    lsnmf = nimfa.Lsnmf(V, seed='random_vcol', rank=2 * x, max_iter=100)
    lsnmf_fit = lsnmf()
    consensus_matrix.append(np.array(lsnmf.consensus()))
    a = lsnmf.coph_cor()
    print(a)
    cophenetic_coeff.append(a)

print('Rss: %5.4f' % lsnmf_fit.fit.rss())
print('Evar: %5.4f' % lsnmf_fit.fit.evar())
print('K-L divergence: %5.4f' % lsnmf_fit.distance(metric='kl'))
print('Sparseness, W: %5.4f, H: %5.4f' % lsnmf_fit.fit.sparseness())
