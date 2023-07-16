import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, confusion_matrix


def fit_svm(train_embs, train_labels):
    train_embs = np.concatenate(train_embs)
    train_labels = np.concatenate(train_labels)

    svm_model = SVC(decision_function_shape='ovo')
    svm_model.fit(train_embs, train_labels)
    train_acc = svm_model.score(train_embs, train_labels)

    return svm_model, train_acc


def evaluate_svm(svm, val_embs, val_labels):

    val_preds = []
    for vid_embs in val_embs:
        vid_preds = svm.predict(vid_embs)
        val_preds.append(vid_preds)

    # concatenate labels and preds in one array
    val_preds = np.concatenate(val_preds)
    val_labels = np.concatenate(val_labels)

    # calculate accuracy and confusion matrix
    val_acc = accuracy_score(val_labels, val_preds)
    conf_mat = confusion_matrix(val_labels, val_preds)

    return val_acc, conf_mat


def evaluate_phase_classification(ckpt_step, train_embs, train_labels, val_embs, val_labels, act_name, CONFIG,  writer=None, verbose=False):

    for frac in CONFIG.EVAL.CLASSIFICATION_FRACTIONS:
        N_Vids = max(1, int(len(train_embs) * frac))
        embs = train_embs[:N_Vids]
        labs = train_labels[:N_Vids]

        if verbose:
            print(f'Fraction = {frac}, Total = {len(train_embs)}, Used = {len(embs)}')

        svm_model, train_acc = fit_svm(embs, labs)
        val_acc, conf_mat = evaluate_svm(svm_model, val_embs, val_labels)

        print('\n-----------------------------')
        print('Fraction: ', frac)
        print('Train-Acc: ', train_acc)
        print('Val-Acc: ', val_acc)
        print('Conf-Mat: ', conf_mat)


        writer.add_scalar(f'classification/train_{act_name}_{frac}', train_acc, global_step=ckpt_step)
        writer.add_scalar(f'classification/val_{act_name}_{frac}', val_acc, global_step=ckpt_step)
        
        print(f'classification/train_{act_name}_{frac}', train_acc, f"global_step={ckpt_step}")
        print(f'classification/val_{act_name}_{frac}', val_acc, f"global_step={ckpt_step}")

    return train_acc, val_acc


def _compute_ap(val_embs, val_labels):
    results = []
    for k in [5, 10, 15]:
        nbrs = NearestNeighbors(n_neighbors=k).fit(val_embs)
        distances, indices = nbrs.kneighbors(val_embs)
        vals = []
        for i in range(val_embs.shape[0]):
            a = np.array([val_labels[i]] * k)
            b = indices[i]
            b = np.array([val_labels[k] for k in b])
            val = (a==b).sum()/k
            vals.append(val)

        results.append(np.mean(vals))
    return results

def compute_ap(videos, labels):
    ap5, ap10, ap15 = 0, 0, 0
    for v,l in zip(videos, labels):
        a5, a10, a15 = _compute_ap(v,l)
        ap5 += a5
        ap10 += a10
        ap15 += a15
    ap5 /= len(videos)
    ap10 /= len(videos)
    ap15 /= len(videos)
    return [ap5, ap10, ap15]