# Propagate labels
def convert_label_list(label_list, max_seq_len):
  labels = []
  curr_label = 0
  for i in range(max_seq_len):
    if i > label_list[curr_label]:
      curr_label += 1
    labels.append(curr_label)
  return labels


def fit_svm_model(train_embs, train_labels):
  """Fit a SVM classifier."""
  svm_model = SVC(decision_function_shape='ovo', verbose=2)
  svm_model.fit(train_embs, train_labels)
  train_acc = svm_model.score(train_embs, train_labels)
  print('Label propagation model accuracy:', train_acc)
  print('If this is too low, propagation will not work properly.')
  return svm_model

def propagate_labels(embs, labels):
  train_embs = []
  train_labels = []
  for video_id in labels:
    max_frame_id = max(labels[video_id])
    train_embs.extend(embs[video_id][:max_frame_id])
    train_labels.extend(convert_label_list(labels[video_id],
                                           max_frame_id))
  model = fit_svm_model(train_embs, train_labels)

  propagated_labels = []
  for video_id in range(len(embs)):
    pred_labels = model.predict(embs[video_id])
    propagated_labels.append(pred_labels)
  return propagated_labels