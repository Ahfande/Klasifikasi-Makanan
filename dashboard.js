const API_BASE_URL = "http://10.37.137.162:5000";
console.log("Api Server: ", API_BASE_URL);

const fileInput = document.getElementById("fileInput");
const PilihFile = document.getElementById("PilihFile");
const areaInput = document.getElementById("areaInput");
const DeteksiMakanan = document.getElementById("KonfirmasiMakanan");
const ResetMakanan = document.getElementById("ResetMakanan");
const loadingIndicator = document.getElementById("loadingIndicator");
const imagePreview = document.getElementById("imagePreview");
const errorMessage = document.getElementById("errorMessage");
const noDeteksi = document.getElementById("noDeteksi");
const TabelContainer = document.getElementById("TabelContainer");
const hasilDeteksi = document.getElementById("hasilDeteksi");
const hasilAkurasi = document.getElementById("hasilAkurasi");
const informasiGizi = document.getElementById("informasiGizi");

// Pilih file
areaInput.addEventListener("click", () => {
  console.log("Klik area upload");
  fileInput.click();
});

PilihFile.addEventListener("click", (e) => {
  e.stopPropagation(); 
  console.log("Klik button pilih file");
  fileInput.click();
});

ResetMakanan.addEventListener("click", () => {
  console.log("sip");
  resetAll();
});

fileInput.addEventListener("change", function (e) {
  console.log("File dipilih:", this.files);

  if (this.files && this.files[0]) {
    const file = this.files[0];
    console.log("File info:", file.name, file.type, file.size);

    if (!file.type.match("image.*")) {
      showError("Silakan pilih file gambar (JPG, PNG, JPEG)");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      showError("Ukuran file terlalu besar. Maksimal 5MB");
      return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      console.log("File berhasil dibaca, menampilkan preview");
      imagePreview.src = e.target.result;
      imagePreview.style.display = "block";

      if (noDeteksi) noDeteksi.style.display = "none";

      resetResults();

      hideError();
    };

    reader.onerror = function () {
      console.error("Gagal membaca file");
      showError("Gagal membaca file. Silakan coba lagi.");
    };

    reader.readAsDataURL(file);
  }
});

areaInput.addEventListener("dragover", (e) => {
  e.preventDefault();
  areaInput.style.backgroundColor = "rgba(98, 129, 65, 0.1)";
  console.log("Drag over area");
});

areaInput.addEventListener("dragleave", () => {
  areaInput.style.backgroundColor = "";
  console.log("Drag leave area");
});

areaInput.addEventListener("drop", (e) => {
  e.preventDefault();
  areaInput.style.backgroundColor = "";
  console.log("File di-drop");

  if (e.dataTransfer.files.length) {
    const file = e.dataTransfer.files[0];
    console.log("File dropped:", file.name);

    if (!file.type.match("image.*")) {
      showError("Silakan drop file gambar (JPG, PNG, JPEG)");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      showError("Ukuran file terlalu besar. Maksimal 5MB");
      return;
    }

    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;

    const changeEvent = new Event("change");
    fileInput.dispatchEvent(changeEvent);
  }
});

function showError(message) {
  if (errorMessage) {
    errorMessage.style.display = "block";
    errorMessage.querySelector("#errorText").textContent = message;
  }
}

function hideError() {
  if (errorMessage) errorMessage.style.display = "none";
}

function resetResults() {
  if (hasilDeteksi) hasilDeteksi.textContent = "";
  if (hasilAkurasi) hasilAkurasi.textContent = "";
  if (informasiGizi) informasiGizi.innerHTML = "";
  if (TabelContainer) TabelContainer.style.display = "none";
  if (noDeteksi) noDeteksi.style.display = "block";
}

function showLoading() {
  if (loadingIndicator) {
    loadingIndicator.style.display = "flex";
  }
}

function hideLoading() {
  if (loadingIndicator) {
    loadingIndicator.style.display = "none";
  }
}

function resetAll() {
  fileInput.value = "";
  imagePreview.src = "";
  imagePreview.style.display = "none";
  loadingIndicator.style.display = "none";
  resetResults();
}

function formatNutritionTable(nutritionData) {
  const kode_tkpi = nutritionData.kode_tkpi || "TKPI-NOT-FOUND";

  const nutritionRows = [
    {
      komponen: "Energi",
      jumlah: nutritionData.energi || 0,
      satuan: "kkal",
      kode: kode_tkpi,
    },
    {
      komponen: "Protein",
      jumlah: nutritionData.protein || 0,
      satuan: "g",
      kode: kode_tkpi,
    },
    {
      komponen: "Lemak",
      jumlah: nutritionData.lemak || 0,
      satuan: "g",
      kode: kode_tkpi,
    },
    {
      komponen: "Karbohidrat",
      jumlah: nutritionData.karbohidrat || 0,
      satuan: "g",
      kode: kode_tkpi,
    },
  ];

  return nutritionRows;
}

if (DeteksiMakanan) {
  DeteksiMakanan.addEventListener("click", async () => {
    console.log("=== Tombol Deteksi diklik ===");

    if (!fileInput.files || !fileInput.files[0]) {
      showError("Silakan pilih gambar makanan terlebih dahulu");
      return;
    }

    const file = fileInput.files[0];
    console.log(
      "File yang dipilih:",
      file.name,
      "Size:",
      file.size,
      "Type:",
      file.type,
    );

    if (!file.type.match("image.*")) {
      showError("Silakan pilih file gambar (JPG, PNG, JPEG)");
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      showError("Ukuran file terlalu besar. Maksimal 5MB");
      return;
    }

    showLoading();
    hideError();

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append("user_id", "skripsi_user");

      console.log("Mengirim request ke:", `${API_BASE_URL}/api/upload`);
      console.log("FormData entries:");
      for (let pair of formData.entries()) {
        console.log(pair[0] + ": " + pair[1]);
      }

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000);

      const response = await fetch(`${API_BASE_URL}/api/upload`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      console.log("Response status:", response.status, response.statusText);

      const contentType = response.headers.get("content-type");
      if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        console.error("Response bukan JSON:", text.substring(0, 200));
        throw new Error(
          `Server error: ${response.status} ${response.statusText}`,
        );
      }

      const result = await response.json();
      console.log("Response dari server:", result);

      hideLoading();

      if (result.success) {
        console.log("✅ Deteksi berhasil!");
        console.log("Makanan:", result.detection.food_name);
        console.log("Confidence:", result.detection.confidence + "%");
        console.log("Data gizi:", result.nutrition);

        if (hasilDeteksi) {
          hasilDeteksi.textContent = result.detection.food_name;
          console.log("Update hasilDeteksi:", result.detection.food_name);
        }

        if (hasilAkurasi) {
          const confidence = result.detection.confidence;
          let confidenceIcon = "";
          let confidenceClass = "";
          
          hasilAkurasi.innerHTML = `${confidenceIcon} ${confidence}%`;
          hasilAkurasi.className = confidenceClass;
          console.log("Update Confidence:", confidence + "%");
        }

        // ========== TAMPILKAN PERFORMANCE MODEL ==========
        const performaModelElement = document.getElementById("performaModel");
        if (performaModelElement && result.model_evaluation) {
          const metrics = result.model_evaluation;        
          const performaText = `Akurasi: ${metrics.accuracy}% | Presisi: ${metrics.precision}% | Recall: ${metrics.recall}% | F1: ${metrics.f1_score}%`;
          
          performaModelElement.innerHTML = performaText;
          performaModelElement.style.fontSize = "12px";
          performaModelElement.style.color = "#628141";
          performaModelElement.style.display = "block";
          performaModelElement.style.marginTop = "5px";
          
          console.log("Update Performa Model:", performaText);
        } else if (performaModelElement) {
          performaModelElement.innerHTML = "Data performa model tidak tersedia";
          performaModelElement.style.color = "#999";
        }

        if (noDeteksi) {
          noDeteksi.style.display = "none";
          console.log("Sembunyikan noDeteksi");
        }

        if (TabelContainer) {
          TabelContainer.style.display = "block";
          console.log("Tampilkan TabelContainer");
        }

        if (informasiGizi) {
          const nutritionRows = formatNutritionTable(result.nutrition);
          informasiGizi.innerHTML = "";
          nutritionRows.forEach((row) => {
            const tr = document.createElement("tr");

            const tdKomponen = document.createElement("td");
            tdKomponen.textContent = row.komponen;

            const tdJumlah = document.createElement("td");
            tdJumlah.textContent = row.jumlah;

            const tdSatuan = document.createElement("td");
            tdSatuan.textContent = row.satuan;

            const tdKode = document.createElement("td");
            tdKode.textContent = row.kode;

            tr.appendChild(tdKomponen);
            tr.appendChild(tdJumlah);
            tr.appendChild(tdSatuan);
            tr.appendChild(tdKode);

            informasiGizi.appendChild(tr);
          });

          console.log("Tabel gizi diisi dengan", nutritionRows.length, "baris");
        }

        if (result.model_evaluation) {
          console.log("📊 Evaluasi Model:");
          console.log(`  Akurasi: ${result.model_evaluation.accuracy}%`);
          console.log(`  Presisi: ${result.model_evaluation.precision}%`);
          console.log(`  Recall: ${result.model_evaluation.recall}%`);
          console.log(`  F1-Score: ${result.model_evaluation.f1_score}%`);
        }

        if (result.image_url && imagePreview) {
          const fullImageUrl = `${API_BASE_URL}${result.image_url}`;
          imagePreview.src = fullImageUrl;
          console.log("Update image preview dengan:", fullImageUrl);
        }
      } else {
        const errorMsg = result.error || "Gagal melakukan deteksi";
        console.error("Server error:", errorMsg);
        showError(errorMsg);
      }
    } catch (error) {
      console.error("Fetch error:", error);
      hideLoading();

      if (error.name === "AbortError") {
        showError("Timeout: Server terlalu lama merespon. Coba lagi.");
      } else if (
        error.message.includes("NetworkError") ||
        error.message.includes("Failed to fetch")
      ) {
        showError(
          "Tidak dapat terhubung ke server. Pastikan backend berjalan di " +
            API_BASE_URL,
        );
      } else {
        showError("Terjadi kesalahan: " + error.message);
      }
    }
  });
}