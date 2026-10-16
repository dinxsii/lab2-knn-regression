from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

OUT = Path(__file__).resolve().parent
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
SEED = 18
rng = np.random.default_rng(SEED)
X = rng.uniform(0, 1000, (1000, 1))
y = np.sin(X[:, 0] / 100) + X[:, 0] / 1000
y = y + rng.normal(0, 0.2, 1000)
ids = np.arange(1000)
tr, te = train_test_split(ids, test_size=0.2, random_state=SEED)
sx, sy = MinMaxScaler(), MinMaxScaler()
Xtr = sx.fit_transform(X[tr])
Xte = sx.transform(X[te])
ytr = sy.fit_transform(y[tr, None]).ravel()
Xn = sx.transform(X)
yn = sy.transform(y[:, None]).ravel()
ks = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51, 75, 101, 151]
rows, models = [], {}
for k in ks:
    model = KNeighborsRegressor(n_neighbors=k, weights='uniform', p=2)
    model.fit(Xtr, ytr)
    pred = sy.inverse_transform(model.predict(Xte)[:, None]).ravel()
    train_pred = sy.inverse_transform(model.predict(Xtr)[:, None]).ravel()
    mse = mean_squared_error(y[te], pred)
    rows.append([k, mean_squared_error(y[tr], train_pred), mse,
                 np.sqrt(mse), mean_absolute_error(y[te], pred),
                 r2_score(y[te], pred)])
    models[k] = model
best = min(rows, key=lambda row: row[2])
kbest = int(best[0])
pred = sy.inverse_transform(models[kbest].predict(Xte)[:, None]).ravel()
residual = y[te] - pred

def save_csv(name, header, records):
    with (OUT / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(records)

split = np.full(1000, 'train', dtype='<U5')
split[te] = 'test'
save_csv('data.csv', ['id', 'x', 'y', 'split'],
         zip(ids, X[:, 0], y, split))
save_csv('normalized_data.csv', ['id', 'x_norm', 'y_norm', 'split'],
         zip(ids, Xn[:, 0], yn, split))
save_csv('metrics.csv', ['k', 'train_MSE', 'test_MSE', 'RMSE', 'MAE', 'R2'], rows)
save_csv('predictions.csv', ['id', 'x', 'y_true', 'y_pred', 'residual'],
         zip(te, X[te, 0], y[te], pred, residual))
summary = dict(best_k=kbest, MSE=best[2], RMSE=best[3], MAE=best[4],
               R2=best[5], x_min=float(sx.data_min_[0]),
               x_max=float(sx.data_max_[0]), y_min=float(sy.data_min_[0]),
               y_max=float(sy.data_max_[0]), residual_mean=float(residual.mean()))
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
plt.style.use('dark_background')
plt.rcParams.update({'figure.facecolor': '#000000', 'axes.facecolor': '#000000',
                     'font.size': 11, 'axes.grid': True, 'grid.alpha': 0.2})

def finish(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=200, facecolor='#000000')
    plt.close()

grid = np.linspace(0, 1000, 1000).reshape(-1, 1)
truth = np.sin(grid[:, 0] / 100) + grid[:, 0] / 1000
plt.figure(figsize=(9, 4.5))
plt.scatter(X[tr, 0], y[tr], s=10, alpha=.55, label='Навчальні: 800')
plt.scatter(X[te, 0], y[te], s=15, c='#ffbd66', label='Тестові: 200')
plt.plot(grid[:, 0], truth, c='#6cf7b4', lw=2, label='Функція без шуму')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(fontsize=9)
plt.title('Згенеровані дані та розподіл вибірки')
finish('01_data.png')
plt.figure(figsize=(9, 4.2))
plt.scatter(Xn[tr, 0], yn[tr], s=10, alpha=.55, label='Навчальні')
plt.scatter(Xn[te, 0], yn[te], s=15, c='#ffbd66', label='Тестові')
plt.xlabel('Нормалізована ознака x'); plt.ylabel('Нормалізована ціль y')
plt.title('Масштабування MinMaxScaler'); plt.legend()
finish('02_normalized.png')
a = np.array(rows)
plt.figure(figsize=(9, 4.2))
plt.plot(a[:, 0], a[:, 1], 'o-', label='MSE навчальної вибірки')
plt.plot(a[:, 0], a[:, 2], 'o-', label='MSE тестової вибірки', c='#ffbd66')
plt.axvline(kbest, c='#6cf7b4', ls='--', label=f'Найкраще k = {kbest}')
plt.xlabel('Кількість сусідів k'); plt.ylabel('MSE'); plt.legend(fontsize=9)
plt.title('Порівняння якості KNN-регресорів')
finish('03_metrics.png')
plt.figure(figsize=(9, 4.5))
plt.scatter(X[te, 0], y[te], s=14, c='#ffbd66', alpha=.65, label='Тестові дані')
plt.plot(grid[:, 0], truth, '--', c='white', lw=1.4, label='Функція без шуму')
for k, color in [(1, '#9671ce'), (kbest, '#6cf7b4'), (151, '#ff6767')]:
    pg = sy.inverse_transform(models[k].predict(sx.transform(grid))[:, None]).ravel()
    plt.plot(grid[:, 0], pg, c=color, lw=1.5, alpha=.85, label=f'k = {k}')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(fontsize=9)
plt.title('Вплив кількості сусідів на прогноз')
finish('04_regression.png')
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].scatter(y[te], pred, s=17, c='#6cf7b4', alpha=.7)
lo, hi = min(y[te].min(), pred.min()), max(y[te].max(), pred.max())
ax[0].plot([lo, hi], [lo, hi], '--', color='white')
ax[0].set(xlabel='Фактичне y', ylabel='Прогноз y', title=f'Найкраща модель k = {kbest}')
ax[1].scatter(pred, residual, s=17, c='#ffbd66', alpha=.7)
ax[1].axhline(0, color='white', ls='--')
ax[1].set(xlabel='Прогноз y', ylabel='Залишок y − прогноз', title='Аналіз залишків')
finish('05_diagnostics.png')
log = ['LAB 2 | KNN REGRESSION', 'Samples: 1000 | Train: 800 | Test: 200',
       f'Random seed: {SEED}', '', ' k     Train MSE   Test MSE    RMSE       MAE        R2']
log += [f'{r[0]:3d}    ' + '   '.join(f'{v:.6f}' for v in r[1:]) for r in rows]
log += ['', f'Best k = {kbest}', f'MSE = {best[2]:.6f} | RMSE = {best[3]:.6f}',
        f'MAE = {best[4]:.6f} | R2 = {best[5]:.6f}']
text = '\n'.join(log)
print(text)
(OUT / 'run_log.txt').write_text(text + '\n', encoding='utf-8')
fig = plt.figure(figsize=(10, 5.5), facecolor='black')
fig.text(.035, .95, text, va='top', family='DejaVu Sans Mono', fontsize=11,
         color='#e5e5e5', linespacing=1.5)
plt.savefig(FIG / '06_output.png', dpi=200, facecolor='black')
plt.close()
