mp = pd.read_csv('/content/marketing_product.csv', delimiter=';')
mdp = pd.read_csv('/content/marketing_dealerprice.csv', delimiter=';')
mpdk = pd.read_csv('/content/marketing_productdealerkey.csv', delimiter=';')
train_mdp, tst_mdp = train_test_split(mdp, test_size=0.3)
ld_mp = mp.to_dict('records')
ld_mdp_train = train_mdp.to_dict('records')
ld_mdp_tst = tst_mdp.to_dict('records')
ld_mpdk = mpdk.to_dict('records')
%time res = match(ld_mp, ld_mdp_train, ld_mpdk, ld_mdp_tst)
df_res = pd.DataFrame(res)
df_res.head()

test = tst_mdp.merge(mpdk, how='left', left_on='product_key',
               right_on='key').loc[:, ['product_key', 'key', 'product_id']]
test = pd.concat([test, df_res], axis=1)
def recall_k(k):
    for i in range(test.shape[0]):
        if test.loc[i, 'product_id'] in test.loc[i, [str(t) for t in range(1, k+1)]].values:
            test.loc[i, 'recall@5'] = 1
        else:
            test.loc[i, 'recall@5'] = 0
recall_k(5)
print('Accuracy : ', sum(test['recall@5'])/test.shape[0])
