# BIST ADAPTIVE ML v3.5.2 — GitHub Paper Automation

Bu paket mevcut adaptif XGBoost sistemini **paper trading** için GitHub Actions'a taşır. Gerçek emir göndermez.

## Kilitli çekirdek
- 350 BIST hissesi
- son 730 gün eğitim
- 120 günlük recency half-life
- her 5 işlem gününde yeniden eğitim
- TP +%5 / SL -%5 / maksimum 5 işlem günü
- Top-3 gösterilir, yalnız #1 PRIMARY paper sinyali olur
- komisyon + slippage dahil paper muhasebesi

## Otomasyon
Workflow iş günlerinde **18:27 ve 19:07 Europe/Istanbul** saatlerinde çalışır. İkinci koşu veri gecikmesine karşı yedektir; aynı gün PRIMARY kilidi ikinci sinyal oluşmasını önler.

- `schedule` => PRIMARY
- Actions > Run workflow > `preview` => yalnız görüntüleme, state değiştirmez
- Actions > Run workflow > `primary` => manuel kalıcı paper çalışması

PRIMARY, Yahoo son günlük barının tarihi Türkiye'deki gün ile eşleşmiyorsa yeni sinyal yazmaz. Bu mekanizma tatil/veri gecikmesinde yanlış güne yeni PRIMARY yazılmasını önler.

## Kurulum
1. GitHub'da **Private** repo oluştur.
2. Bu ZIP'in içeriğini repo köküne yükle.
3. Default branch üzerinde `.github/workflows/bist_daily.yml` bulunduğundan emin ol.
4. Repo > Settings > Actions > General bölümünde workflow'un yazma iznine engel olmadığını kontrol et. Workflow ayrıca `permissions: contents: write` ister.
5. Actions sekmesinden ilk çalıştırmayı `preview` ile yap.
6. Preview sağlamsa otomatik schedule'a bırak.

## Mevcut state
Paket, konuşmada paylaşılan en güncel state ile hazırlanmıştır:
- model bloğu: 2026-08-10
- model trained-through: 2026-08-07
- IEYHO 2026-08-07 PRIMARY -> 2026-08-10 açılıştan OPEN
- PASEU 2026-08-10 PRIMARY -> sonraki gerçek açılışı bekliyor

## Dosyalar
- `main.py`: motor
- `state/`: XGBoost model + 5 günlük blok meta
- `data/`: Top-3 ve paper kayıtları
- `output/latest_report.md`: son çalışma özeti
- `.github/workflows/bist_daily.yml`: zamanlayıcı
- `tests/`: hızlı güvenlik testleri

## Kritik kural
Backtest çekirdeği değişmedikçe `PAPER_VERSION`, `FEATURES`, TP/SL/HOLD veya XGBoost parametrelerini değiştirme. Değişiklik yapılırsa eski performans rakamları yeni sistem için geçerli kabul edilmemelidir.
