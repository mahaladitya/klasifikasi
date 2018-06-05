library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('hasil real count di rutan kpk dimenangkan anies sandi.', 'poster bertuliskan jika anies kalah akan ada muslim revolusi.', 'di rutan kpk ahok djarot tidak mendapat suara.', 'tidak terdapat foto ktp peserta pemilu yang memiliki kesamaan nama dan nik.',  'saat rekapitulasi di daerah tebet ahok djarot mendapat tambahan suara', 'Heboh Jackie Chan Masuk Islam.', 'Bahaya Cool Fever yang menempel di Dahi', 'Pesan Berantai Korban Tragedi Mina', 'Campuran Udang dan Vitamin C Dapat Menjadi Racun yang Berbahaya.', 'Berita Pancing Hujan Menggunakan Air Garam di Baskom',
          'ahok djarot yang memenangkan suara di rutan kpk.', 'tidak ada poster ancaman jika anies kalah', 'ahok dan djarot memperoleh suara terbanyak yaitu 61 suara.', 'foto ktp peserta pemilu memiliki kesamaan nama dan nik.', 'tidak ada penambahan suara saat rekapitulasi di daerah tebet', 'Tidak ada konfirmasi resmi dari pihak Jackie Chan mengenai kepindahan dia masuk Islam.', 'dokter di Indonesia dan Malaysia menyangkal berita tersebut dan mengatakan tidak benar.', 'Korban tersebut memang korban tragedi Mina namun yang terjadi pada tahun 2004.', 'keracunan akibat mengkonsumsi udang dan vitamin C dalam waktu berdekatan tidaklah benar.', 'Air garam dalam baskom itu tidak bisa membuat uap dan menyebabkan hujan turun.')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))

train
colnames(train)[ncol(train)] <- 'y'
train
train <- as.data.frame(train)
train$y <- as.factor(train$y)

# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('server kpu dikendalikan oleh ainun najib yang mendukung jokowi.', 'server kpu berpusat di singapura.', 'suara ahok djarot bertambah 200.', 'server kpu di tingkat kelurahan dibobol.', 'ktp ganda dengan foto yang sama.',
           'ainun najib yang menggagas pilkada.', 'server kpu di imam bonjol.', 'suara ahok djarot tetap sama yaitu 61 suara.', 'tidak ada server di tingkat kelurahan.', 'foto ktp yang sama ternyata dipalsukan.')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)
