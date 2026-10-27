import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#среднее значение - done
#стандартное отклонение - done
#минимум - done
#максимум- done
#различные квантили
def paint_graph_by_column(data, column):

    plt.plot(data)
    plt.title(column)




    mean = data.mean()
    max = data.max()
    min = data.min()
    std = data.std()
    quater = data.quantile(0.25)
    half = data.quantile(0.5)
    big_quater = data.quantile(0.75)

    #print(dscrb)
    #print(dscrb["mean"])

    plt.axhline(y=mean, color="gray", linestyle="--", label="среднее")

    plt.axhline(y=max, color="red", linestyle="--", label="макс")

    plt.axhline(y=min, color="green", linestyle="--", label="мин")

    plt.axvline(x = min-std, color = "blue", linestyle = ":")
    plt.axvline(x = min-std, color = "blue", linestyle = ":")

    plt.axhline(y = quater, color = "purple", linestyle = "--", label="25%")
    plt.axhline(y=half, color="purple", linestyle="--", label="50%")
    plt.axhline(y=big_quater, color="purple", linestyle="--", label="75%")
    plt.legend()
    plt.show()

#Разделите данные на обучающий и тестовый наборы данных.

#Реализуйте линейную регрессию с использованием метода наименьших квадратов без использования сторонних библиотек, кроме NumPy и Pandas (для использования коэффициентов использовать библиотеки тоже нельзя). Использовать минимизацию суммы квадратов разностей между фактическими и предсказанными значениями для нахождения оптимальных коэффициентов.

#Постройте три модели с различными наборами признаков.

#Для каждой модели проведите оценку производительности, используя метрику коэффициент детерминации, чтобы измерить, насколько хорошо модель соответствует данным.

#Сравните результаты трех моделей и сделайте выводы о том, какие признаки работают лучше всего для каждой модели.

#https://habr.com/ru/articles/479398/
def do_regression(data, *args):
    y = np.array(data[["median_house_value"]], dtype=np.float64)   # (n, 1)
    X = np.array(data[list(args)], dtype=np.float64)               # (n, k)
    X = np.column_stack([np.ones(len(X)), X])                      # столбец единиц для w0

    w = np.linalg.solve(X.T @ X, X.T @ y)                          # XᵀX w = Xᵀy

    y_pred = X @ w
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    print("w:", w.ravel())
    print("R2:", r2)

    return w, r2

def draw_test(test_data, w, *args):
    y = np.array(test_data[["median_house_value"]], dtype=np.float64)
    X = np.array(test_data[list(args)], dtype=np.float64)
    X = np.column_stack([np.ones(len(X)), X])

    y_pred = X @ w
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2)

    print("R2 (тест):", r2)

    plt.scatter(y, y_pred, alpha=0.5, label="тестовые данные")
    lims = [y.min(), y.max()]
    plt.plot(lims, lims, color="red", linestyle="--", label="идеал y = ŷ")
    plt.xlabel("фактическая цена")
    plt.ylabel("предсказанная цена")
    plt.title(f"{', '.join(args)}: R2 = {r2:.3f}")
    plt.legend()
    plt.show()

    return r2

def main():
    df = pd.read_csv(
        'california_housing_train.csv'
        , sep = ','
        ,encoding ='utf-8'
    )
    #for i in df.columns:
    #    paint_graph_by_column(df[i], i)

    #100 тестовых значений
    learn_cnt = 50
    learn_data  = df.sample(n=learn_cnt)
    print(learn_data.head())

    #print(df.desяЯ знаю::cribe())
    test_cnt = 100
    test_df = df.sample(n=test_cnt)
    # ----------------------- Модель 1 ------------------------
    w1,r1 = do_regression(learn_data, "median_income")
    t1 = draw_test(test_df, w1, "median_income")
    # ----------------------- Модель 2 ------------------------
    w2,r2 = do_regression(learn_data, "median_income", "housing_median_age", "total_rooms")
    t2 = draw_test(test_df, w2, "median_income", "housing_median_age", "total_rooms")
    # ----------------------- Модель 3 ------------------------
    w3,r3 = do_regression(learn_data, "median_income", "housing_median_age", "total_rooms",
              "total_bedrooms", "population", "households")
    t3 = draw_test(test_df, w3, "median_income", "housing_median_age", "total_rooms",
                   "total_bedrooms", "population", "households")

    print("R2 train:", r1, r2, r3)
    print("R2 test: ", t1, t2, t3)
if __name__ == '__main__':


    main()