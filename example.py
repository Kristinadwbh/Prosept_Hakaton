mp = pd.read_csv('/content/marketing_product.csv', delimiter=';')
mdp = pd.read_csv('/content/marketing_dealerprice.csv', delimiter=';')
mpdk = pd.read_csv('/content/marketing_productdealerkey.csv', delimiter=';')

# деление данных на обучение и тест
train_mdp, tst_mdp = train_test_split(mdp, test_size=0.3)

# перевод фрэймов в список словарей
ld_mp = mp.to_dict('records')
ld_mdp_train = train_mdp.to_dict('records')
ld_mdp_tst = tst_mdp.to_dict('records')
ld_mpdk = mpdk.to_dict('records')

# обучение модели
%time model = matching_training(ld_mp, ld_mdp_train, ld_mpdk)

# получение предсказания
%time res = matching_predict(ld_mp, ld_mdp_tst, model)

df_res = pd.DataFrame(res)
df_res.head()

# Получение метрики recall@k (или accuracy)
test = tst_mdp.merge(mpdk,
                     how='left',
                     left_on='product_key',
                     right_on='key').loc[:, ['product_key',
                                             'key',
                                             'product_id']]
test = pd.concat([test, df_res], axis=1)
test = test.loc[~test.product_id.isnull()]
test.reset_index(drop=True, inplace=True)

def recall_k(k, test):
    for i in range(test.shape[0]):
        if test.loc[i, 'product_id'] in test.loc[
            i, [str(t) for t in range(1, k + 1)]].values:
            test.loc[i, f'recall@{str(k)}'] = 1
        else:
            test.loc[i, f'recall@{str(k)}'] = 0
    return k

k = recall_k(df_res.shape[1], test)
print('Accuracy : ', sum(test[f'recall@{str(k)}']) / test.shape[0])


