import streamlit as st
import streamlit.components.v1 as components
import os

# 파일 경로 지정 (같은 디렉토리라면)
file_path = os.path.join("pages", "03_내분점.html")

# HTML 파일 읽기
with open(file_path, "r", encoding="utf-8") as f:
    html_code = f.read()

# 임베딩
components.html(html_code, height=600, scrolling=True)

<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>선분 내분점 탐구 도구 (모바일 최적화)</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <!-- Chosen Palette: Calm Harmony -->
    <!-- Application Structure Plan: 모바일 우선 접근법(Mobile-First Approach)을 적용하여 구조를 설계했습니다. 모든 요소는 기본적으로 작은 화면에 맞춰 세로로 쌓이며, 화면이 커짐에 따라 다단 레이아웃으로 확장됩니다. 1) 상호작용의 핵심인 차트를 최상단에 배치. 2) 그 아래에 컨트롤 패널을 배치. 3) 마지막으로 학습 결과물인 탐구 활동지를 배치하여 '탐색 -> 조작 -> 정리'의 자연스러운 흐름을 유지하면서도 모바일 사용성을 극대화했습니다. -->
    <!-- Visualization & Content Choices: 
        - 보고서 정보: 좌표평면, 점 A/B, 내분점 P, 수선의 발, 내분비 m:n, 탐구 활동 질문.
        - 목표: 모바일 환경에서도 내분점 원리를 원활하게 탐구하고, 학습 결과를 이미지로 저장.
        - 시각화/표현: Chart.js Scatter 플롯 사용. 화면 크기에 따라 차트 높이와 폰트 크기가 동적으로 조절됩니다.
        - 상호작용: 기존의 마우스 드래그 기능에 더해, 모바일 사용자를 위한 터치(touchstart, touchmove, touchend) 이벤트를 추가하여 터치스크린에서의 조작성을 향상시켰습니다.
        - 정당성: 모바일 우선 레이아웃과 터치 이벤트 지원은 다양한 기기에서의 접근성과 사용성을 보장합니다. 이는 학생들이 언제 어디서든 학습 도구를 효과적으로 활용할 수 있도록 합니다.
        - 라이브러리/방법: Chart.js, MathJax, html2canvas, Tailwind CSS, Vanilla JS.
    -->
    <!-- CONFIRMATION: NO SVG graphics used. NO Mermaid JS used. -->
    <style>
        body {
            font-family: 'Pretendard', sans-serif;
            background-color: #f8f7f4;
            color: #383734;
        }
        .chart-container {
            position: relative;
            width: 100%;
            height: 55vh; /* 모바일 기본 높이 */
            max-height: 450px; /* 모바일 최대 높이 */
            cursor: default;
        }
        @media (min-width: 768px) { /* 태블릿 이상 */
            .chart-container {
                height: 65vh;
                max-height: 600px;
            }
        }
        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none; appearance: none;
            width: 20px; height: 20px;
            background: #a88b79; cursor: pointer; border-radius: 50%;
        }
        input[type="range"]::-moz-range-thumb {
            width: 20px; height: 20px;
            background: #a88b79; cursor: pointer; border-radius: 50%;
        }
        .draggable { cursor: grab; }
        .dragging { cursor: grabbing; }
    </style>
    <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css" />
</head>
<body class="antialiased">
    <div class="container mx-auto p-4 sm:p-6 lg:p-8">
        
        <header class="text-center mb-6">
            <h1 class="text-2xl sm:text-3xl md:text-4xl font-bold text-[#5a504b]">선분 내분점 탐구 도구</h1>
            <p class="mt-2 text-base sm:text-lg text-[#6e6a66]">점을 직접 움직이거나 좌표와 비율을 조절하며 내분점의 원리를 탐색해 보세요.</p>
        </header>

        <main class="w-full">
            <div class="bg-white/70 p-2 sm:p-4 rounded-2xl shadow-sm border border-gray-200/80 mb-8">
                <div id="chart-wrapper" class="chart-container">
                    <canvas id="coordPlane"></canvas>
                </div>
            </div>
            
            <div class="bg-white/70 p-6 rounded-2xl shadow-sm border border-gray-200/80 mb-10">
                <h2 class="text-2xl font-bold mb-4 text-[#5a504b]">설정 및 결과</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <div>
                        <h3 class="text-lg font-semibold mb-2 text-[#a88b79]">점 A 좌표</h3>
                        <div class="flex items-center gap-3">
                            <label for="ax" class="text-sm font-medium text-gray-700 w-8">X:</label>
                            <input type="number" id="ax" value="-6" step="0.1" class="block w-full rounded-md border-gray-300 shadow-sm p-2">
                        </div>
                        <div class="flex items-center gap-3 mt-2">
                            <label for="ay" class="text-sm font-medium text-gray-700 w-8">Y:</label>
                            <input type="number" id="ay" value="2" step="0.1" class="block w-full rounded-md border-gray-300 shadow-sm p-2">
                        </div>
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold mb-2 text-[#a88b79]">점 B 좌표</h3>
                        <div class="flex items-center gap-3">
                            <label for="bx" class="text-sm font-medium text-gray-700 w-8">X:</label>
                            <input type="number" id="bx" value="8" step="0.1" class="block w-full rounded-md border-gray-300 shadow-sm p-2">
                        </div>
                        <div class="flex items-center gap-3 mt-2">
                            <label for="by" class="text-sm font-medium text-gray-700 w-8">Y:</label>
                            <input type="number" id="by" value="7" step="0.1" class="block w-full rounded-md border-gray-300 shadow-sm p-2">
                        </div>
                    </div>
                    <div class="md:col-span-2 lg:col-span-1">
                        <h3 class="text-lg font-semibold mb-2 text-[#a88b79]">내분 비율 (m : n)</h3>
                        <div>
                            <label for="m" class="text-sm font-medium text-gray-700">m = <span id="m_val" class="font-bold">3</span></label>
                            <input type="range" id="m" min="1" max="10" value="3" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
                        </div>
                        <div class="mt-2">
                            <label for="n" class="text-sm font-medium text-gray-700">n = <span id="n_val" class="font-bold">2</span></label>
                            <input type="range" id="n" min="1" max="10" value="2" class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer">
                        </div>
                    </div>
                </div>
                <div class="mt-6 pt-6 border-t border-gray-200 text-center">
                    <h3 class="text-xl font-semibold mb-2 text-[#5a504b]">내분점 P의 좌표</h3>
                    <div class="text-lg scale-90 sm:scale-100">$$P = \left( \frac{mx_B + nx_A}{m+n}, \frac{my_B + ny_A}{m+n} \right)$$</div>
                    <p class="text-xl mt-2 font-mono font-bold text-[#a88b79]" id="p_coords"></p>
                </div>
            </div>
        </main>

        <section id="worksheet" class="bg-white/70 p-6 rounded-2xl shadow-sm border border-gray-200/80">
            <form id="submission-form" onsubmit="return false;">
                <div class="flex flex-col sm:flex-row sm:justify-between sm:items-center mb-6">
                    <h2 class="text-2xl font-bold text-[#5a504b] mb-4 sm:mb-0">탐구 활동지</h2>
                    <div class="flex items-center gap-2">
                        <label for="student_info" class="font-semibold flex-shrink-0">학번/이름:</label>
                        <input type="text" id="student_info" name="student_info" required class="rounded-md border-gray-300 shadow-sm p-2 w-full sm:w-40">
                    </div>
                </div>
                <div class="space-y-6">
                    <div>
                        <label for="q1" class="font-semibold block mb-2">1. 슬라이더 m의 값을 n보다 훨씬 크게 만들어보세요 (예: m=10, n=1). 내분점 P는 점 A와 점 B 중 어느 쪽에 더 가까워지나요?</label>
                        <textarea id="q1" name="q1" rows="2" required class="w-full rounded-md border-gray-300 shadow-sm p-2"></textarea>
                    </div>
                    <div>
                        <label for="q2" class="font-semibold block mb-2">2. 반대로 n의 값을 m보다 훨씬 크게 만들어보세요 (예: m=1, n=10). 이번엔 점 P가 어느 쪽에 가까워지나요?</label>
                        <textarea id="q2" name="q2" rows="2" required class="w-full rounded-md border-gray-300 shadow-sm p-2"></textarea>
                    </div>
                    <div>
                        <label for="q3" class="font-semibold block mb-2">3. m과 n의 값을 똑같이 맞춰보세요 (예: m=5, n=5). 점 P는 선분 AB의 어떤 점이 되나요? 이 점을 무엇이라고 부르죠?</label>
                        <textarea id="q3" name="q3" rows="2" required class="w-full rounded-md border-gray-300 shadow-sm p-2"></textarea>
                    </div>
                    <div>
                        <label for="q4" class="font-semibold block mb-2">4. 점 P가 움직일 때, x축 위의 점(\(P_x\))은 점 A, B의 x좌표를 잇는 선분을 \(m:n\)으로 내분하는 것처럼 보이나요? y축 위의 점(\(P_y\))은 어떤가요?</label>
                        <textarea id="q4" name="q4" rows="2" required class="w-full rounded-md border-gray-300 shadow-sm p-2"></textarea>
                    </div>
                    <div>
                        <label for="q5" class="font-semibold block mb-2">5. 점 A와 B의 좌표를 다양하게 바꿔보며 위 활동들을 반복해 보세요. 어떤 규칙을 발견할 수 있었나요? 자유롭게 서술해 보세요.</label>
                        <textarea id="q5" name="q5" rows="3" required class="w-full rounded-md border-gray-300 shadow-sm p-2"></textarea>
                    </div>
                </div>
                <div class="mt-8 text-center">
                    <button id="download-btn" type="button" class="bg-[#a88b79] text-white font-bold py-3 px-8 rounded-lg hover:bg-[#937766] transition-colors shadow-md">이미지로 저장하기</button>
                </div>
            </form>
        </section>

    </div>

    <script>
        document.addEventListener('DOMContentLoaded', () => {
            const canvas = document.getElementById('coordPlane');
            const ctx = canvas.getContext('2d');
            const chartWrapper = document.getElementById('chart-wrapper');

            const inputs = {
                ax: document.getElementById('ax'), ay: document.getElementById('ay'),
                bx: document.getElementById('bx'), by: document.getElementById('by'),
                m: document.getElementById('m'), n: document.getElementById('n'),
            };

            const displays = {
                m_val: document.getElementById('m_val'), n_val: document.getElementById('n_val'),
                p_coords: document.getElementById('p_coords'),
            };

            const chart = new Chart(ctx, {
                type: 'scatter',
                data: {
                    datasets: [
                        { label: '선분 AB', data: [], borderColor: 'rgba(168, 139, 121, 0.5)', borderWidth: 2, showLine: true, pointRadius: 0, order: 5 },
                        { label: '점 A', data: [], backgroundColor: '#6b8a7a', pointRadius: 8, pointHoverRadius: 10, order: 1 },
                        { label: '점 B', data: [], backgroundColor: '#6b8a7a', pointRadius: 8, pointHoverRadius: 10, order: 2 },
                        { label: '내분점 P', data: [], backgroundColor: '#d3755b', pointRadius: 9, pointHoverRadius: 11, order: 0 },
                        { label: '수선', data: [], borderColor: 'rgba(168, 139, 121, 0.6)', borderWidth: 1.5, borderDash: [4, 4], showLine: true, pointRadius: 0, order: 4 },
                        { label: '수선', data: [], borderColor: 'rgba(168, 139, 121, 0.6)', borderWidth: 1.5, borderDash: [4, 4], showLine: true, pointRadius: 0, order: 4 },
                        { label: '수선의 발', data: [], backgroundColor: 'rgba(168, 139, 121, 0.8)', pointRadius: 5, pointHoverRadius: 7, order: 3 },
                        { label: 'X-Axis', data: [], borderColor: 'rgba(54, 54, 54, 0.7)', borderWidth: 1.5, showLine: true, pointRadius: 0, type: 'line', order: 6 },
                        { label: 'Y-Axis', data: [], borderColor: 'rgba(54, 54, 54, 0.7)', borderWidth: 1.5, showLine: true, pointRadius: 0, type: 'line', order: 6 }
                    ]
                },
                options: {
                    responsive: true, maintainAspectRatio: false,
                    plugins: { legend: { display: false }, tooltip: {
                        callbacks: { label: (c) => `${c.dataset.label}: (${c.raw.x.toFixed(2)}, ${c.raw.y.toFixed(2)})` }
                    }},
                    scales: {
                        x: { type: 'linear', position: 'bottom', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { stepSize: 2 } },
                        y: { type: 'linear', position: 'left', grid: { color: 'rgba(0, 0, 0, 0.05)' }, ticks: { stepSize: 2 } }
                    }
                }
            });

            let draggedPoint = null;

            function updateVisualization() {
                const ax = parseFloat(inputs.ax.value); const ay = parseFloat(inputs.ay.value);
                const bx = parseFloat(inputs.bx.value); const by = parseFloat(inputs.by.value);
                const m = parseInt(inputs.m.value, 10); const n = parseInt(inputs.n.value, 10);

                displays.m_val.textContent = m;
                displays.n_val.textContent = n;

                const px = (m * bx + n * ax) / (m + n);
                const py = (m * by + n * ay) / (m + n);
                
                displays.p_coords.textContent = `P = (${px.toFixed(2)}, ${py.toFixed(2)})`;

                const points = {
                    A: { x: ax, y: ay }, B: { x: bx, y: by },
                    P: { x: px, y: py }, Px: { x: px, y: 0 }, Py: { x: 0, y: py }
                };

                chart.data.datasets[0].data = [points.A, points.B];
                chart.data.datasets[1].data = [points.A];
                chart.data.datasets[2].data = [points.B];
                chart.data.datasets[3].data = [points.P];
                chart.data.datasets[4].data = [points.P, points.Px];
                chart.data.datasets[5].data = [points.P, points.Py];
                chart.data.datasets[6].data = [points.Px, points.Py];

                const allX = [ax, bx, 0]; const allY = [ay, by, 0];
                const minX = Math.min(...allX) - 2; const maxX = Math.max(...allX) + 2;
                const minY = Math.min(...allY) - 2; const maxY = Math.max(...allY) + 2;

                chart.data.datasets[7].data = [{x: minX, y: 0}, {x: maxX, y: 0}];
                chart.data.datasets[8].data = [{x: 0, y: minY}, {x: 0, y: maxY}];

                chart.options.scales.x.min = minX; chart.options.scales.x.max = maxX;
                chart.options.scales.y.min = minY; chart.options.scales.y.max = maxY;

                chart.update('none');
            }

            function getChartCoordinates(event) {
                const rect = canvas.getBoundingClientRect();
                const clientX = event.type.startsWith('touch') ? event.touches[0].clientX : event.clientX;
                const clientY = event.type.startsWith('touch') ? event.touches[0].clientY : event.clientY;
                const x = clientX - rect.left;
                const y = clientY - rect.top;
                return {
                    x: chart.scales.x.getValueForPixel(x),
                    y: chart.scales.y.getValueForPixel(y)
                };
            }

            function handleDragStart(event) {
                const elements = chart.getElementsAtEventForMode(event, 'point', { intersect: true }, true);
                if (elements.length > 0) {
                    const firstElement = elements[0];
                    if (firstElement.datasetIndex === 1) draggedPoint = 'A';
                    else if (firstElement.datasetIndex === 2) draggedPoint = 'B';
                    
                    if (draggedPoint) {
                        chartWrapper.classList.add('dragging');
                        if (event.cancelable) event.preventDefault();
                    }
                }
            }

            function handleDragMove(event) {
                if (draggedPoint) {
                    if (event.cancelable) event.preventDefault();
                    const coords = getChartCoordinates(event);
                    if (draggedPoint === 'A') {
                        inputs.ax.value = coords.x.toFixed(1);
                        inputs.ay.value = coords.y.toFixed(1);
                    } else if (draggedPoint === 'B') {
                        inputs.bx.value = coords.x.toFixed(1);
                        inputs.by.value = coords.y.toFixed(1);
                    }
                    updateVisualization();
                } else {
                    const elements = chart.getElementsAtEventForMode(event, 'point', { intersect: true }, true);
                    chartWrapper.classList.toggle('draggable', elements.length > 0 && (elements[0].datasetIndex === 1 || elements[0].datasetIndex === 2));
                }
            }

            function handleDragEnd() {
                draggedPoint = null;
                chartWrapper.classList.remove('dragging');
                chartWrapper.classList.remove('draggable');
            }

            canvas.addEventListener('mousedown', handleDragStart);
            canvas.addEventListener('mousemove', handleDragMove);
            canvas.addEventListener('mouseup', handleDragEnd);
            canvas.addEventListener('mouseout', handleDragEnd);
            
            canvas.addEventListener('touchstart', handleDragStart, { passive: false });
            canvas.addEventListener('touchmove', handleDragMove, { passive: false });
            canvas.addEventListener('touchend', handleDragEnd);
            canvas.addEventListener('touchcancel', handleDragEnd);

            Object.values(inputs).forEach(input => {
                input.addEventListener('input', updateVisualization);
            });
            
            document.getElementById('download-btn').addEventListener('click', function() {
                const worksheetElement = document.getElementById('worksheet');
                const studentInfo = document.getElementById('student_info').value.trim() || '이름없음';
                const fileName = `선분내분점_탐구활동지_${studentInfo}.png`;
                
                const originalBg = worksheetElement.style.backgroundColor;
                worksheetElement.style.backgroundColor = '#FFFFFF';

                html2canvas(worksheetElement, {
                    useCORS: true,
                    scale: 2,
                    onclone: (doc) => {
                        return new Promise((resolve) => {
                            if(window.MathJax) {
                                MathJax.typesetPromise([doc.body]).then(resolve);
                            } else {
                                resolve();
                            }
                        });
                    }
                }).then(canvas => {
                    worksheetElement.style.backgroundColor = originalBg;
                    const image = canvas.toDataURL('image/png');
                    const link = document.createElement('a');
                    link.href = image;
                    link.download = fileName;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }).catch(err => {
                    console.error('이미지 변환 중 오류가 발생했습니다.', err);
                    worksheetElement.style.backgroundColor = originalBg;
                    alert('이미지 저장에 실패했습니다. 콘솔을 확인해주세요.');
                });
            });

            updateVisualization();
        });
    </script>
</body>
</html>
