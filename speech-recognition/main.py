import database
import vectors
import hmm


hmms = database.LoadAllHmms()

def one():
  vector_size, num_vectors, num_states = map(int, raw_input("unesite velicinu mfcc vektora, broj input vektora i broj stanja u hmm(ako vec  postoji hmm ovaj broj ce se ignorisati)\n").split())
  print "unesite %d redova od po %d brojeva" % (num_vectors,  vector_size)
  
  base_vectors = []
  gauss_sizes = []
  for i in xrange(0, num_vectors):
    base_vectors.append(map(float, raw_input().split()))
    x = int(raw_input("unesite koliko gausovskih vektora da se dodatno generise uz poslednji vektor\n"))
    gauss_sizes.append(x)

  print 'Unos (+ generisani vektori)'
  generated_vectors = []
  for i in xrange(0, num_vectors):
      generated_vectors.append(base_vectors[i])
      print base_vectors[i]
      genvec = vectors.GenerateGaussian(base_vectors[i], gauss_sizes[i])
      for v in genvec:
        print v
      print ''
      generated_vectors.extend(genvec)

  if vector_size == 2:
    vectors.Draw2D(generated_vectors)

  label = str(raw_input("Unesite labelu(ime pod kojim ce se snimiti) ovaj hmm\n")).strip()

  found = False
  for i in xrange(0, len(hmms)):
    if hmms[i].label == label:
      found = True
      hmms[i].AppendMfccVectors(generated_vectors)
      hmms[i].KMeans()
      database.SaveHmm(hmms[i])
      break

  if found == False:
    h = hmm.Hmm(num_states, vector_size, label)
    h.AppendMfccVectors(generated_vectors)
    h.KMeans()
    database.SaveHmm(h)
    hmms.append(h)

  print ''
  main()

def two():
  vector_size, num_vectors = map(int, raw_input("unesite velicinu mfcc vektora i broj input vektora\n").split())
  print "unesite %d redova od po %d brojeva" % (num_vectors,  vector_size)
  
  base_vectors = []
  gauss_sizes = []
  for i in xrange(0, num_vectors):
    base_vectors.append(map(float, raw_input().split()))
    x = int(raw_input("unesite koliko gausovskih vektora da se dodatno generise uz poslednji vektor\n"))
    gauss_sizes.append(x)

  print 'Unos (+ generisani vektori)'
  generated_vectors = []
  for i in xrange(0, num_vectors):
      generated_vectors.append(base_vectors[i])
      print base_vectors[i]
      genvec = vectors.GenerateGaussian(base_vectors[i], gauss_sizes[i])
      for v in genvec:
        print v
      print ''
      generated_vectors.extend(genvec)

  if vector_size == 2:
    vectors.Draw2D(generated_vectors)

  for h in hmms:
    print "%.7lf %s" % (h.CalculateScore(generated_vectors), h.label)

  print ''
  main()

def main():
  print '1. Dodavanje novog hmm-a u bazu.'
  print '2. Pretraga'
  print '3. Izlaz'
  cmd = int(raw_input())

  if cmd == 1:
    one()
  elif cmd == 2:
    two()

main()
