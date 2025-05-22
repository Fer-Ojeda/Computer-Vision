// app.js (Versión Corregida y Mejorada)
document.addEventListener('DOMContentLoaded', () => {
    // Selectores de elementos (asegúrate que coinciden con interfaz.html)
    const seccionSeleccionAnaquel = document.getElementById('seccionSeleccionAnaquel');
    const seccionCargaImagen = document.getElementById('seccionCargaImagen');
    const seccionResultados = document.getElementById('seccionResultados');

    const anaquelButtons = document.querySelectorAll('.anaquel-button');
    const imageUploadInput = document.getElementById('imageUpload');
    const imagenParaRecortarEl = document.getElementById('imagenParaRecortar');
    const editorImagenContainer = document.getElementById('editorImagenContainer');
    const btnRecortar = document.getElementById('btnRecortar');
    const btnOmitirRecorte = document.getElementById('btnOmitirRecorte');
    const loader = document.getElementById('loader');
    const estadoProcesamientoBackendEl = document.getElementById('estadoProcesamientoBackend');
    const areaResultados = document.getElementById('areaResultados');
    const imagenResultadosEl = document.getElementById('imagenResultados');
    const instruccionesItemsUl = document.getElementById('instruccionesItems');
    
    const btnNuevoAnalisisResultados = document.getElementById('btnNuevoAnalisis'); // El de la sección resultados
    const btnExportarResultados = document.getElementById('btnExportarResultados');
    const btnVolverInicioResultados = document.getElementById('btnVolverInicioResultados');

    const mensajeErrorResultadosEl = document.getElementById('mensajeErrorResultados');
    const btnRegresarSeleccion = document.getElementById('btnRegresarSeleccion');


    let cropper = null;
    let anaquelSeleccionado = null;
    let imagenSubidaTemporalURL = null;
    let imagenOriginalNombre = "";

    // --- Navegación entre secciones ---
    function mostrarSeccion(idSeccion) {
        document.querySelectorAll('main section').forEach(sec => sec.classList.remove('active-section'));
        document.getElementById(idSeccion).classList.add('active-section');
        
        // Mostrar/ocultar botón de regresar en carga de imagen
        if (idSeccion === 'seccionCargaImagen') {
            btnRegresarSeleccion.style.display = 'inline-block';
        } else {
            btnRegresarSeleccion.style.display = 'none';
        }
    }

    // --- Lógica de Selección de Anaquel ---
    anaquelButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            anaquelSeleccionado = e.currentTarget.dataset.anaquel;
            document.getElementById('anaquelSeleccionadoTitulo').textContent = `- Anaquel ${anaquelSeleccionado}`;
            resetCargaImagen();
            mostrarSeccion('seccionCargaImagen');
        });
    });

    btnRegresarSeleccion.addEventListener('click', () => {
        mostrarSeccion('seccionSeleccionAnaquel');
        anaquelSeleccionado = null; // Resetea la selección
        document.getElementById('anaquelSeleccionadoTitulo').textContent = "";
    });

    function resetCargaImagen() {
        imageUploadInput.value = null;
        editorImagenContainer.style.display = 'none';
        if (cropper) {
            cropper.destroy();
            cropper = null;
        }
        imagenParaRecortarEl.src = '#';
        if (imagenSubidaTemporalURL) {
            URL.revokeObjectURL(imagenSubidaTemporalURL);
            imagenSubidaTemporalURL = null;
        }
        imagenOriginalNombre = "";
    }

    // --- Lógica de Carga y Recorte de Imagen ---
    imageUploadInput.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file && file.type.startsWith("image/")) {
            imagenOriginalNombre = file.name;
            const reader = new FileReader();
            reader.onload = (e) => {
                imagenParaRecortarEl.src = e.target.result;
                editorImagenContainer.style.display = 'block';
                if (cropper) {
                    cropper.destroy();
                }
                cropper = new Cropper(imagenParaRecortarEl, {
                    aspectRatio: 0,
                    viewMode: 1,
                    dragMode: 'move',
                    autoCropArea: 0.85, // Un poco más grande por defecto
                    responsive: true,
                    background: false,
                    modal: true,
                    cropBoxResizable: true,
                    cropBoxMovable: true,
                });
            };
            reader.readAsDataURL(file);
        } else if (file) {
            alert("Por favor, selecciona un archivo de imagen válido (JPEG, PNG, etc.).");
            resetCargaImagen();
        }
    });
    
    document.getElementById('btnZoomInCrop')?.addEventListener('click', () => cropper?.zoom(0.1));
    document.getElementById('btnZoomOutCrop')?.addEventListener('click', () => cropper?.zoom(-0.1));
    document.getElementById('btnResetCrop')?.addEventListener('click', () => cropper?.reset());

    btnRecortar.addEventListener('click', () => {
        if (!cropper) return alert("Carga una imagen para recortar.");
        cropper.getCroppedCanvas({
            // Opciones para mejorar calidad si es necesario
            // minWidth: 1024, 
            // minHeight: 768,
            // imageSmoothingEnabled: true,
            // imageSmoothingQuality: 'high',
        }).toBlob((blob) => {
            if (imagenSubidaTemporalURL) URL.revokeObjectURL(imagenSubidaTemporalURL);
            imagenSubidaTemporalURL = URL.createObjectURL(blob);
            enviarImagenAlBackend(blob, true);
        }, 'image/jpeg', 0.9); // Calidad 0.9 para JPEG
    });

    btnOmitirRecorte.addEventListener('click', () => {
        const file = imageUploadInput.files[0];
        if (file) {
            if (imagenSubidaTemporalURL) URL.revokeObjectURL(imagenSubidaTemporalURL);
            imagenSubidaTemporalURL = URL.createObjectURL(file);
            enviarImagenAlBackend(file, false);
        } else {
            alert("Por favor, selecciona una imagen primero.");
        }
    });

    // --- Comunicación con Backend y Resultados ---
    async function enviarImagenAlBackend(imagenBlob, fueRecortada) {
        mostrarSeccion('seccionResultados');
        loader.style.display = 'block';
        areaResultados.style.display = 'none';
        mensajeErrorResultadosEl.style.display = 'none';
        document.getElementById('anaquelResultadosTitulo').textContent = `- Anaquel ${anaquelSeleccionado}`;
        estadoProcesamientoBackendEl.textContent = "Preparando imagen...";

        const formData = new FormData();
        const nombreArchivoParaEnvio = imagenOriginalNombre || `anaquel_img_${new Date().getTime()}.jpg`;
        formData.append('imagen', imagenBlob, nombreArchivoParaEnvio);
        formData.append('anaquel_id', anaquelSeleccionado);

         try {
            estadoProcesamientoBackendEl.textContent = `Enviando imagen al servidor (Anaquel ${anaquelSeleccionado})...`;
            
            const response = await fetch('http://127.0.0.1:5000/api/analizar-anaquel', { 
                method: 'POST',
                body: formData,
            });

            estadoProcesamientoBackendEl.textContent = "Esperando respuesta del análisis...";
            
            const responseText = await response.text();
            let resultados;
            try {
                resultados = JSON.parse(responseText);
            } catch (e) {
                console.error("Error al parsear JSON de respuesta:", e);
                console.error("Texto de respuesta recibido:", responseText);
                throw new Error(`Respuesta inesperada del servidor. Estado: ${response.status}. Verifica la consola del servidor Flask.`);
            }

            if (!response.ok) {
                throw new Error(`Error del servidor: ${response.status} - ${resultados.error || response.statusText}`);
            }
            
            resultados.imagen_procesada_url = imagenSubidaTemporalURL; // Usar la imagen local para mostrar
            mostrarResultados(resultados);

        } catch (error) {
            console.error("Error al procesar la imagen:", error);
            mensajeErrorResultadosEl.textContent = `Hubo un error: ${error.message}. Por favor, intenta de nuevo. Revisa la consola del navegador y del servidor Flask para más detalles.`;
            mensajeErrorResultadosEl.style.display = 'block';
            loader.style.display = 'none';
        }
    }
    
    // Función de simulación (para cuando el backend no está listo)
    function generarResultadosSimulados() {
        // ... (Mantener la función de simulación que te di antes, es útil para pruebas)
        const esAnaquel1 = anaquelSeleccionado === "1";
        const numResultados = Math.floor(Math.random() * 8) + 5;
        const resultados = {
            imagen_procesada_url: imagenSubidaTemporalURL,
            instrucciones: []
        };
        const productosAnaquel1 = ["Atun Dolores", "Nutralat Forte", "La Lechera", "Sabritas", "Cheetos", "Arroz Posada", "Frijol Sierra"];
        const productosAnaquel2 = ["Papel Regio", "Detergente Azalea", "Aceite Mennen", "NAN Optipro", "Agua Bioleve", "Pañales Tikytin"];
        const productosBase = esAnaquel1 ? productosAnaquel1 : productosAnaquel2;
        const tipos = ["OK", "ERROR", "FALTANTE", "EXTRA"];
        for (let i = 0; i < numResultados; i++) {
            const tipo = tipos[Math.floor(Math.random() * tipos.length)];
            const productoEsperado = productosBase[Math.floor(Math.random() * productosBase.length)];
            let productoDetectado = null; let accionSugerida = ""; let claseCss = "";
            switch (tipo) {
                case "OK": productoDetectado = productoEsperado; accionSugerida = "Todo en orden."; claseCss = "instruccion-ok"; break;
                case "ERROR": productoDetectado = productosBase[Math.floor(Math.random() * productosBase.length)]; while(productoDetectado === productoEsperado) { productoDetectado = productosBase[Math.floor(Math.random() * productosBase.length)];} accionSugerida = `Corregir. Reemplazar '${productoDetectado}' por '${productoEsperado}'.`; claseCss = "instruccion-error"; break;
                case "FALTANTE": accionSugerida = `Colocar '${productoEsperado}'.`; claseCss = "instruccion-faltante"; break;
                case "EXTRA": productoDetectado = productosBase[Math.floor(Math.random() * productosBase.length)]; accionSugerida = `Retirar '${productoDetectado}'. Slot debería estar vacío o con otro producto.`; claseCss = "instruccion-extra"; break;
            }
            resultados.instrucciones.push({
                id_item: `fila${Math.ceil(i/3)}_pos${(i%3)+1}_${tipo.toLowerCase()}`,
                fila_planograma: `Fila ${Math.ceil(i/4) +1}`, posicion_en_fila: `Posición ${(i % 4) + 1}`,
                producto_esperado: productoEsperado, producto_detectado: productoDetectado,
                status: tipo, accion: accionSugerida, clase_css: claseCss
            });
        }
        return resultados;
    }

    function mostrarResultados(resultados) {
        loader.style.display = 'none';
        areaResultados.style.display = 'block';
        imagenResultadosEl.src = resultados.imagen_procesada_url || imagenSubidaTemporalURL;

        instruccionesItemsUl.innerHTML = '';
        if (resultados.instrucciones && resultados.instrucciones.length > 0) {
            resultados.instrucciones.forEach(item => {
                const li = document.createElement('li');
                li.classList.add(item.clase_css || 'instruccion-generica');
                
                let contenidoHtml = `<span class="instruccion-header">${item.status} en ${item.fila_planograma || 'N/A'}, ${item.posicion_en_fila || 'N/A'}</span>`;
                if (item.producto_esperado) {
                    contenidoHtml += `<span class="instruccion-detalle">Esperado: <strong>${item.producto_esperado}</strong></span>`;
                }
                if (item.producto_detectado) {
                    contenidoHtml += `<span class="instruccion-detalle">Detectado: <strong>${item.producto_detectado}</strong></span>`;
                }
                contenidoHtml += `<span class="instruccion-accion">Acción: ${item.accion}</span>`;
                li.innerHTML = contenidoHtml;

                const btnFeedback = document.createElement('button');
                btnFeedback.classList.add('btn-reportar-problema');
                btnFeedback.innerHTML = '⚑';
                btnFeedback.title = "Reportar problema con esta detección";
                btnFeedback.onclick = () => abrirModalFeedback(item);
                li.appendChild(btnFeedback);
                instruccionesItemsUl.appendChild(li);
            });
        } else {
            instruccionesItemsUl.innerHTML = '<li>No se generaron instrucciones específicas o el anaquel está perfecto.</li>';
        }
        // resetCargaImagen(); // No resetear aquí para que el usuario vea la imagen que subió. Se resetea al inicio.
    }
    
    // --- Lógica de Modales (Feedback y Ayuda) ---
    const modalFeedback = document.getElementById('modalFeedback');
    const formFeedback = document.getElementById('formFeedback');
    const feedbackItemNombreEl = document.getElementById('feedbackItemNombre');
    const feedbackItemIdInput = document.getElementById('feedbackItemId');

    window.abrirModalFeedback = (item) => {
        feedbackItemIdInput.value = item.id_item || `ItemDesconocido_${new Date().getTime()}`;
        let nombreReferencia = item.producto_esperado || item.producto_detectado || "Item no identificado";
        feedbackItemNombreEl.textContent = `${nombreReferencia} (en ${item.fila_planograma || 'Fila desc.'}, ${item.posicion_en_fila || 'Pos. desc.'})`;
        formFeedback.reset();
        modalFeedback.style.display = 'block';
    };
    window.cerrarModalFeedback = () => {
        modalFeedback.style.display = 'none';
    };

    formFeedback.addEventListener('submit', async (e) => {
        e.preventDefault();
        const formData = new FormData(formFeedback);
        const feedbackData = {
            item_id: feedbackItemIdInput.value,
            tipo_error: formData.get('tipoError'),
            comentario: formData.get('comentarioFeedback'),
            anaquel: anaquelSeleccionado,
            nombre_imagen: imagenOriginalNombre || "No disponible"
        };
        
        console.log("Enviando feedback:", feedbackData);
        try {
            const response = await fetch('http://127.0.0.1:5000/api/registrar-feedback', { 
                method: 'POST', 
                body: JSON.stringify(feedbackData), 
                headers: {'Content-Type': 'application/json'} 
            });
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || "Error al enviar feedback");
            }
            alert("Feedback enviado correctamente. ¡Gracias!");
        } catch (err) {
            console.error("Error enviando feedback:", err);
            alert(`Error al enviar feedback: ${err.message}. Verifica la consola.`);
        }
        cerrarModalFeedback();
    });

    const modalAyuda = document.getElementById('modalAyuda');
    document.getElementById('btnAyuda').addEventListener('click', () => {
        modalAyuda.style.display = 'block';
    });
    window.onclick = function(event) {
        if (event.target == modalFeedback) cerrarModalFeedback();
        if (event.target == modalAyuda) modalAyuda.style.display = "none";
    }

    // --- Botones de Navegación Global ---
    function irAlInicio() {
        anaquelSeleccionado = null;
        resetCargaImagen(); // Limpia la imagen y el cropper
        if (imagenSubidaTemporalURL) { // Asegurarse de liberar el blob si existe
            URL.revokeObjectURL(imagenSubidaTemporalURL);
            imagenSubidaTemporalURL = null;
        }
        mostrarSeccion('seccionSeleccionAnaquel');
    }
    
    btnNuevoAnalisisResultados.addEventListener('click', irAlInicio);
    btnVolverInicioResultados.addEventListener('click', irAlInicio);


    // --- Exportación a PDF ---
    btnExportarResultados.addEventListener('click', async () => {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF({
            orientation: 'p', // portrait
            unit: 'mm',
            format: 'a4'
        });
        const margin = 15; // Aumentar margen
        const pageWidth = pdf.internal.pageSize.getWidth();
        const pageHeight = pdf.internal.pageSize.getHeight();
        const contentWidth = pageWidth - 2 * margin;
        let currentY = margin;

        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(18);
        pdf.text(`Resultados del Análisis - Anaquel ${anaquelSeleccionado}`, pageWidth / 2, currentY, { align: 'center' });
        currentY += 12;

        if (imagenResultadosEl.src && imagenResultadosEl.src !== '#' && imagenResultadosEl.src !== 'placeholder-anaquel.png') {
            try {
                 // Usar html2canvas para capturar la imagen y cualquier anotación que pudieras añadir con canvas
                const canvasElement = await html2canvas(document.getElementById('contenedorImagenResultados'), {
                    useCORS: true, // Si la imagen es de un origen diferente (aunque aquí es local blob)
                    scale: 2 // Aumentar resolución de captura
                });
                const imgData = canvasElement.toDataURL('image/jpeg', 0.85); // calidad buena

                const imgProps = pdf.getImageProperties(imgData);
                let imgHeight = (imgProps.height * contentWidth) / imgProps.width;
                let imgWidth = contentWidth;

                if (imgHeight > pageHeight * 0.4) { // Limitar altura de imagen a 40% de la página
                    imgHeight = pageHeight * 0.4;
                    imgWidth = (imgProps.width * imgHeight) / imgProps.height;
                }
                 if (imgWidth > contentWidth) { // Si aún así es muy ancha
                    imgWidth = contentWidth;
                    imgHeight = (imgProps.height * imgWidth) / imgProps.width;
                }


                if (currentY + imgHeight > pageHeight - margin) {
                    pdf.addPage();
                    currentY = margin;
                }
                pdf.addImage(imgData, 'JPEG', margin, currentY, imgWidth, imgHeight);
                currentY += imgHeight + 8;
            } catch (error) {
                console.error("Error al añadir imagen al PDF:", error);
                pdf.setFont("helvetica", "normal");
                pdf.setFontSize(10);
                pdf.setTextColor(255, 0, 0);
                pdf.text("Error: No se pudo cargar la imagen del anaquel para el PDF.", margin, currentY);
                pdf.setTextColor(0, 0, 0);
                currentY += 7;
            }
        }

        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(14);
        pdf.text("Instrucciones de Corrección:", margin, currentY);
        currentY += 7;
        
        const tableData = [];
        const items = instruccionesItemsUl.querySelectorAll('li');
        items.forEach(item => {
            const status = item.querySelector('.instruccion-header')?.textContent.split(' en ')[0] || "N/A";
            const ubicacion = item.querySelector('.instruccion-header')?.textContent.split(' en ')[1] || "N/A";
            const esperado = Array.from(item.querySelectorAll('.instruccion-detalle')).find(el => el.textContent.includes("Esperado:"))?.textContent.replace("Esperado: ", "").trim() || "-";
            const detectado = Array.from(item.querySelectorAll('.instruccion-detalle')).find(el => el.textContent.includes("Detectado:"))?.textContent.replace("Detectado: ", "").trim() || "-";
            const accion = item.querySelector('.instruccion-accion')?.textContent.replace("Acción: ", "").trim() || "N/A";
            tableData.push([status, ubicacion, esperado, detectado, accion]);
        });

        if (tableData.length > 0) {
            pdf.autoTable({
                head: [['Estado', 'Ubicación', 'Esperado', 'Detectado', 'Acción Sugerida']],
                body: tableData,
                startY: currentY,
                margin: { left: margin, right: margin },
                theme: 'grid', // 'striped', 'grid', 'plain'
                headStyles: { fillColor: [52, 152, 219], textColor: [255,255,255], fontStyle: 'bold' }, // Azul primario
                styles: { fontSize: 9, cellPadding: 1.5, overflow: 'linebreak' },
                columnStyles: {
                    0: { cellWidth: 20 }, // Estado
                    1: { cellWidth: 35 }, // Ubicación
                    2: { cellWidth: 35 }, // Esperado
                    3: { cellWidth: 35 }, // Detectado
                    4: { cellWidth: 'auto' } // Acción
                },
                didDrawPage: function (data) {
                    currentY = data.cursor.y + 5; // Actualizar Y para la siguiente sección si es necesario
                }
            });
        } else {
            pdf.setFont("helvetica", "normal");
            pdf.setFontSize(10);
            pdf.text("No hay instrucciones detalladas para mostrar.", margin, currentY);
        }
        
        pdf.save(`analisis_anaquel_${anaquelSeleccionado}_${new Date().toISOString().slice(0,10)}.pdf`);
    });

    // --- Inicio de la aplicación ---
    mostrarSeccion('seccionSeleccionAnaquel');
});