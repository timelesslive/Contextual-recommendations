函数 :getUserFeature,
输入:包含多条用户信息的dataframe ,输出是user_feature数组
1.  user_id 对应于 UserId
2.  user_gender 对应于 Gender
3.  user_age 对应于 Age
4. user_education 对应于 Education
5. user_major 对应于 Major
6.  user_marital 对应于 Marital
7.  user_interest 对应于 Interest
8. history_article_id : bookHistory字段对应的列表取最前面的非0的值,假设为k个,那么就生成一个size=(batch_size, k) 的tesor
9. history_text_feature :根据history_article_id中的每个书籍 id ,从 字典变量book_info中获取的title字段
10. history_categories : 根据history_article_id中的每个书籍 id,从典变量book_info中获取type字段
11. query_article_id :随机从 dim_config的 history_article_id以内的索引中挑选3个id ,但是确保不和dataframe中的 curBook和positiveSample 字段对应的id相等
12. query_text_feature :根据query_article_id中的每个书籍 id ,从 字典变量book_info中获取的title字段
13. query_categories : 根据query_article_id中的每个书籍 id,从典变量book_info中获取type字段