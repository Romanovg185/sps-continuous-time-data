import numpy as np

m = np.loadtxt('intervals_significant_correlation.csv', delimiter=',')
z = m[:-1, 0] - m[1:, 1]
print(np.median(z))

m = np.loadtxt('cbl_results.csv', delimiter=',')
z = m[1:, :] - m[:-1, :]
s = z.flatten()
print(np.nanmedian(s))

m = np.loadtxt('ctx_results.csv', delimiter=',')
z = m[1:, :] - m[:-1, :]
s = z.flatten()
print(np.nanmedian(s))

m_cbl = np.loadtxt('cbl_results.csv', delimiter=',')
m_ctx = np.loadtxt('ctx_results.csv', delimiter=',')
while m_cbl.shape[0] < m_ctx.shape[0]:
    to_stack = np.full((m_cbl.shape[1], 1), np.nan).T
    m_cbl = np.vstack([m_cbl, to_stack])
while m_cbl.shape[0] > m_ctx.shape[0]:
    to_stack = np.full((m_ctx.shape[1], 1), np.nan).T
    m_ctx = np.vstack([m_ctx, to_stack])
m = np.hstack([m_cbl, m_ctx])
z = m[1:, :] - m[:-1, :]
s = z.flatten()
print(np.nanmedian(s))
