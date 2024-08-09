# BilgenAI_TEKNOFEST2024_TTDI-_Dogal-Dil-Isleme-Senaryosu
Bilişim Vadisi ve Teknofest ortaklığında yapılan Doğal Dil İşleme Yarışması, Senaryosu kapsamında oluşturulan tüm kodlar ve verilen bulunduğu repodur. #Acıkhack2024TDDİ

Selenium ve ChromeWebDriver kullanılarak oluşturulan mainŞikayet.py dosyası içindeki kod ile Web Scrapping yöntemi kullanılarak öncelikle https://www.sikayetvar.com/turkcell/internet-paketi adresi üzerinden internet paketleri üzerine olan, 15.07-2024-20.07.2024 tarihleri arasında bulunan şikayetlerden oluşan küçük bir veri seti oluşturuldu. Bu veri seti sikayet-var.csv adlı csv dosyasında bulunmaktadır. Sitenin kaynak kodu üzerinden elde edilen .complaint-layer.card-v3-container css kodu içinde bulunan href linkleri ile veriler arasında gezilerek, .complaint-detail-description değişkeni ile şikayet metnini, .complaint-title değişkeni ile şikayet başlığını alan bu süreç zarfında site sayfasında bulunan cookies ve reklamların engellenmesine yarayan kod içerikleri bulunmaktadır. Try-catch yapılarıyla kod, hataları tespit etme konusunda daha duyarlı bir hale getirilmiştir. Bu kodun çalışması için ChromeDriver.exe dosyasına sahip olmak zorundasınız. Aksi halde WebDriver istenildiği gibi çalışmayacaktır. Ekranı sürekli aşağı kaydırmaya ve veri toplanmasının sağlanması adına kod içerikleri de bulunmaktadır. Daha sonrasında bu veri toplama işlemi https://www.sikayetvar.com/turkcell adresine taşınarak her türlü konuda, daha uzun süre çalışan, daha büyük bir şikayet veri seti oluşturulmaya yarayacak şekilde 21.07-2024-08.07.2024 tarihleri arasında elde edilen verilerle güncellenmiştir. Bu kod üzerinde, bu kapsamda düzenlenenler sayfa linkleri ve sayfa sayıları olmuştur. Bu kod özelinde sayfa sayısı 350'ye yükseltilmiştir.Elde edilen şikayet verileri sikayet-var-full.csv adlı csv dosyasında bulunmaktadır.

Aynı şekilde mainTeşekkür.py birebir aynı kod işlemlerinin aynısını gerçekleştirip sadece şikayet sonrası yapılan geri dönüşlerden etkilenen kullanıcıların yaptığı teşekkür yorumlarını içeren, bunun için de .complaint-detail-description değişkeni yerine .solution-content-body kullanılmıştır ve https://www.sikayetvar.com/turkcell/internet-paketi?resolved=true ve https://www.sikayetvar.com/turkcell?resolved=true adresleri üzerinden  21.07-2024-08.07.2024 günleri arasında elde edilen teşekkür verilerini içermektedir. Daha dar kapsamlı olan veri seti Pozitifsikayet-var.csv iken Pozitifsikayet-var-full.csv ise daha geniş kapsamlı veri setidir. 270 sayfa sayısı bulunmaktadır.

Yarışma sürecinde kullanılan kodların, ideal çalıştırılma sırası utils.py, train.py, nlp.py ve main.py olarak ayarlanmıştır. utils.py dosyası içerisinde diğer modüllerde rahatça kullanabilmek için ayrıca tanımlanan kodları içermektedir. temizle adlı fonksiyon, özel karakterlerin veriden çıkartılması, tüm sayı değerlerinin, anlam karmaşası yaratmaması adına 'NUM' adlı bir karaktere dönüştürülmesi, NLTK kütüphanesi içerisinde bulunan Türkçe durdurma kelimelerinin veriden çıkartılmasını içermektedir. BERT ile tüm modüllerde kullanılabilmek için tokenizer tanımlanmıştır. Aynı zamanda yarışma kapsamında bulunası istenen, bulunması desteklenen varlık tespiti için de bir fonksiyon belirtmiştir. Entity listesi ilk olarak bu fonksiyon içerisinde üretilmeye başlanıp, temizle fonksiyonunu kendi içine çağırmakta ve Zeyrek kütüphanesinin Morph Analyzer özelliğini kullanmaktadır. Bu kapsamda büyük harf içeren, ya da büyük harf ile başlayan sözcükler Entity kabul edilirken, bazı sözcüklerle başlayan ve bazı sözcüklerle biten kelimeler direkt entity olarak tanımlanmıştır. Metinleri vectorize etme özelliğini de bu fonksiyon içinde aktive ederken, train.py çalıştırıldığında oluşacak, ve sonrasında nlp.py dosyasında çağrılacak lstm modeli tahminleri için de değişken ayarlanmıştır. Ayrıca son olarak ön işleme adımları ve Zeyrek'in morfoloji analizi sonucu Unknown yani bilinmeyen olarak belirtilen bazı varlıkları tekrardan decode ederek eski hallerine dönüştürmüştür. Verimizin Turkcell, Vodafone gibi kelimeleri içinde büyük harf olsa da olmasa da entity olarak saymasının sebebi özellikle de bu kod bloğudur. Buna rağmen oluşan Unk sözcükler ise belirlenen sınırın üstünde olup çok daha yabancı kelime olarak nitelendirildiği için ya da özel bir karakterin araya kaçabilmesi sonucu oluşmuştur. Son olarak da utils.py dosyası içinde varlıklara label encoder yapılmasını sağlayan fonksiyon bulunmaktadır. Bu sayede lstm gibi sayısal veriye ihtiyaç duyan modeller daha başarılı çalışabilecektir.
