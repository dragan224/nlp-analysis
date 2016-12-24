import hmm
import os

def LoadSpecificHmm(label_name):
  try:
    model_dir = os.path.join('baza', label_name)
    model_file = os.path.join(model_dir, 'model')
    return hmm.FromFile(model_file)
  except:
    pass

def LoadAllHmms():
  hmms = []
  root = os.listdir('baza')
  for d in root:
    try:
      model_dir = os.path.join('baza', d)
      model_file = os.path.join(model_dir, 'model')
      hmms.append(hmm.FromFile(model_file))
    except:
      pass
  return hmms

def SaveHmm(hmm):
  root_dir = os.path.join('baza', hmm.label)
  try:
    os.mkdir(root_dir)
  except:
    pass
  
  try:
    hmm.WriteToFile(os.path.join(root_dir, 'model'))
  except:
    pass

  i = 1
  for vector_arr in hmm.features:
    try:
      f = open(os.path.join(root_dir, str(i) + '.txt'), "w")
      for vector in vector_arr:
        f.write("%s\n" % vector)
      f.close()
      i += 1
    except:
      pass
