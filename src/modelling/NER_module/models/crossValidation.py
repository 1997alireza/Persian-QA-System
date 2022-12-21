import numpy as np
from src.modelling.NER_module.models.word2vec import load_word2vec


def cross_validate_create_models(slices, model_class, path, x, y, dict, dict2, dict_rev, dict_rev2, iter_num,
                                 word2vec_address=None, timesteps=60, use_gru_cells=False, h_num=128,
                                 add_to_train_x=None, add_to_train_y=None, MTL=False,
                                 ax=None, ay=None, adict2=None, adict_rev2=None, aiter_num=20,
                                 train_on_auxiliary=False):
    for i in range(slices):
        test_start = int((i / slices) * len(x))
        test_end = int(((i + 1) / slices) * len(x))
        # print('test_start', test_start)
        # print('test_end', test_end)
        x_train = np.concatenate((x[:test_start], x[test_end:]), axis=0)
        y_train = np.concatenate((y[:test_start], y[test_end:]), axis=0)
        ax_train = np.concatenate((ax[:test_start], ax[test_end:]), axis=0)
        ay_train = np.concatenate((ay[:test_start], ay[test_end:]), axis=0)
        if add_to_train_x is not None or add_to_train_y is not None:
            x_train = np.concatenate((x_train, add_to_train_x), axis=0)
            y_train = np.concatenate((y_train, add_to_train_y), axis=0)
        if word2vec_address is None:
            if MTL:
                model = model_class(dict, dict2, dict_rev, dict_rev2, adict2, adict_rev2,
                                    path + str(i + 1), iter_nums=iter_num,
                                    timesteps=timesteps, gru=use_gru_cells, num_hiddens=h_num)

            else:
                model = model_class(dict, dict2, dict_rev, dict_rev2, path + str(i + 1), iter_nums=iter_num,
                                    timesteps=timesteps, gru=use_gru_cells, num_hiddens=h_num)
        else:
            w2v = load_word2vec(word2vec_address)
            if MTL:
                model = model_class(dict, dict2, dict_rev, dict_rev2, adict2, adict_rev2, path + str(i + 1),
                                    iter_nums=iter_num,
                                    pretrained_w2v=w2v, timesteps=timesteps, gru=use_gru_cells, num_hiddens=h_num)
            else:
                model = model_class(dict, dict2, dict_rev, dict_rev2, path + str(i + 1), iter_nums=iter_num,
                                    pretrained_w2v=w2v, timesteps=timesteps, gru=use_gru_cells, num_hiddens=h_num)
        if train_on_auxiliary:
            # print("TRAINING ON AUXILIARY")
            model.train(ax_train, ay_train, aiter_num, target=False)
        # print("TRAINING ON TARGET")
        model.train(x_train, y_train, iter_num)

        # sess.close()


def cross_validate_test_models(slices, model_class, path, x, y, dict, dict2, dict_rev, dict_rev2, test_func,
                               word2vec_address=None, precision_recall=False, log_to_file=False, MTL=False, adict2=None,
                               adict_rev2=None, on_target=True, log_tests=False):
    errors = []
    prs = []
    recs = []
    for i in range(slices):
        test_start = int((i / slices) * len(x))
        test_end = int(((i + 1) / slices) * len(x))
        x_test = x[test_start:test_end]
        y_test = y[test_start:test_end]
        if word2vec_address is None:
            if MTL:
                model = model_class(dict, dict2, dict_rev, dict_rev2, adict2, adict_rev2, path + str(i + 1))
            else:
                model = model_class(dict, dict2, dict_rev, dict_rev2, path + str(i + 1))
        else:
            w2v = load_word2vec(word2vec_address)
            if MTL:
                model = model_class(dict, dict2, dict_rev, dict_rev2, adict2, adict_rev2, path + str(i + 1),
                                    pretrained_w2v=w2v)
            else:
                model = model_class(dict, dict2, dict_rev, dict_rev2, path + str(i + 1), pretrained_w2v=w2v)

        model.load_model()
        if precision_recall:
            if not on_target:
                pr, rec = test_func(model, x_test, y_test, dict, adict2, log_to_file=log_to_file, target=False,
                                    log_tests=log_tests)
            else:
                pr, rec = test_func(model, x_test, y_test, dict, dict2, log_to_file=log_to_file, log_tests=log_tests)
            prs.append(pr)
            recs.append(rec)
        else:
            if not on_target:

                err = test_func(model, x_test, y_test, dict, adict2, log_to_file=log_to_file, target=False,
                                log_tests=log_tests)
            else:
                err = test_func(model, x_test, y_test, dict, dict2, log_to_file=log_to_file, log_tests=log_tests)

            errors.append(err)
        # sess.close()
    fscores = [(2 * prs[i] * recs[i]) / (prs[i] + recs[i]) for i in range(len(prs))]
    print("---------- Overall Result ----------")
    if precision_recall:
        print('Precision', "{0:.3f}".format(np.mean(prs)))
        prs = [float("{0:.3f}".format(num)) for num in prs]
        print(prs)
        print('Recall', "{0:.3f}".format(np.mean(recs)))
        recs = [float("{0:.3f}".format(num)) for num in recs]
        print(recs)
        fscores = [float("{0:.3f}".format(num)) for num in fscores]
        print('F Socres', fscores)
        print('Avg. F Score', "{0:.3f}".format(np.mean(fscores)))
    else:
        print('Error Rate', "{0:.3f}".format(np.mean(errors)))
        errors = [float("{0:.3f}".format(num)) for num in errors]
        print(errors)
