document.addEventListener('DOMContentLoaded', function() {

    window.prmLeaderboardData = []; // Global variable to store data
    let currentSortHeader = null;
    let currentSortDirection = null;

    // Configuration for sections and their sub-details
    const sectionConfig = {
        's1': {
            topCellClass: 'prm-s1-top-cell',
            overallColClass: 'prm-s1-overall-col',
            detailColClass: 'prm-s1-details',
            detailKeys: ['S1','NR', 'NCL'],
            detailColspan: 3 // Number of columns when expanded
        },
        's2': {
            topCellClass: 'prm-s2-top-cell',
            overallColClass: 'prm-s2-overall-col',
            detailColClass: 'prm-s2-details',
            detailKeys: ['S2','ES', 'SC', 'DC', 'CI'],
            detailColspan: 5
        },
        's3': {
            topCellClass: 'prm-s3-top-cell',
            overallColClass: 'prm-s3-overall-col',
            detailColClass: 'prm-s3-details',
            detailKeys: ['S3','PS', 'DR', 'MS'],
            detailColspan: 4
        }
    };

    function fetchDataAndLoadTable() {
        fetch('./static/data/prmbench_leaderboard.json')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                window.prmLeaderboardData = data;
                loadTableData(data);
                setupEventListeners();
                window.addEventListener('resize', adjustColumnWidths);
            })
            .catch(error => {
                console.error('Error loading table data:', error);
                const tbody = document.querySelector('#prmbench-table tbody');
                if (tbody) {
                    tbody.innerHTML = `
            <tr>
              <td colspan="15" style="text-align: center; color: red;">
                Error loading data: ${error.message}<br>
                Please ensure the JSON file path is correct and accessible.
              </td>
            </tr>
          `;
                }
            });
    }

    function loadTableData(data) {
        const tbody = document.querySelector('#prmbench-table tbody');
        tbody.innerHTML = '';

        // Calculate styling ranks for all relevant fields
        const ranks = prepareScoresForStyling(data);

        // Determine the overall best model for the gold medal
        const maxOverallScore = Math.max(...data.map(row => parseFloat(row.Overall) || 0));

        data.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.classList.add(row.Class.toLowerCase().replace('-', '_'));
            // console.log(tr)
            let modelNameHtml = row.Source && row.Source.trim() !== ''
                ? `<a href="${row.Source}" target="_blank" class="ext-link">${row.Model}</a>`
                : row.Model;

            // if (parseFloat(row.Overall) === maxOverallScore) {
            //     modelNameHtml += '🥇';
            // }

            tr.innerHTML = `
        <td>${modelNameHtml}</td>
        <td>${applyStyle(row.Overall, ranks.Overall[index])}</td>
<!--        <td>${row.Class || '-'}</td>-->

        <!-- S1 Group -->
        <td class="${sectionConfig.s1.overallColClass}">${applyStyle(row.S1, ranks.S1[index])}</td>
        <td class="hidden ${sectionConfig.s1.detailColClass}">${applyStyle(row.S1, ranks.S1[index])}</td>
        <td class="hidden ${sectionConfig.s1.detailColClass}">${applyStyle(row.NR, ranks.NR[index])}</td>
        <td class="hidden ${sectionConfig.s1.detailColClass}">${applyStyle(row.NCL, ranks.NCL[index])}</td>

        <!-- S2 Group -->
        <td class="${sectionConfig.s2.overallColClass}">${applyStyle(row.S2, ranks.S2[index])}</td>
        <td class="hidden ${sectionConfig.s2.detailColClass}">${applyStyle(row.S2, ranks.S2[index])}</td>
        <td class="hidden ${sectionConfig.s2.detailColClass}">${applyStyle(row.ES, ranks.ES[index])}</td>
        <td class="hidden ${sectionConfig.s2.detailColClass}">${applyStyle(row.SC, ranks.SC[index])}</td>
        <td class="hidden ${sectionConfig.s2.detailColClass}">${applyStyle(row.DC, ranks.DC[index])}</td>
        <td class="hidden ${sectionConfig.s2.detailColClass}">${applyStyle(row.CI, ranks.CI[index])}</td>

        <!-- S3 Group -->
        <td class="${sectionConfig.s3.overallColClass}">${applyStyle(row.S3, ranks.S3[index])}</td>
        <td class="hidden ${sectionConfig.s3.detailColClass}">${applyStyle(row.S3, ranks.S3[index])}</td>
        <td class="hidden ${sectionConfig.s3.detailColClass}">${applyStyle(row.PS, ranks.PS[index])}</td>
        <td class="hidden ${sectionConfig.s3.detailColClass}">${applyStyle(row.DR, ranks.DR[index])}</td>
        <td class="hidden ${sectionConfig.s3.detailColClass}">${applyStyle(row.MS, ranks.MS[index])}</td>
      `;
            tbody.appendChild(tr);
        });

        setTimeout(adjustColumnWidths, 0);
        initializeSorting();
    }

    function setupEventListeners() {
        // Reset button
        document.querySelector('.reset-cell').addEventListener('click', function() {
            resetTable();
        });

        // Top-level clickable headers for S1, S2, S3
        document.querySelector(`.${sectionConfig.s1.topCellClass}`).addEventListener('click', function() { toggleDetails('s1'); });
        document.querySelector(`.${sectionConfig.s2.topCellClass}`).addEventListener('click', function() { toggleDetails('s2'); });
        document.querySelector(`.${sectionConfig.s3.topCellClass}`).addEventListener('click', function() { toggleDetails('s3'); });

        // Sorting headers (only second row headers are sortable)
        const headers = document.querySelectorAll('#prmbench-table thead tr:last-child th.sortable');
        headers.forEach(header => {
            header.addEventListener('click', function() {
                sortTable(this);
            });
        });
    }

    function toggleDetails(sectionName) {
        const sections = ['s1', 's2', 's3']; // All possible sections

        sections.forEach(sec => {
            const config = sectionConfig[sec];
            const topHeaderCell = document.querySelector(`.${config.topCellClass}`); // Top-level S1/S2/S3 header
            const overallHeaderInSecondRow = document.querySelector(`th[data-key="${config.overallKey || sec.toUpperCase()}"]`); // S1/S2/S3 header in second row
            const detailHeaders = document.querySelectorAll(`th.${config.detailColClass}`); // NR, NCL etc. headers in second row

            const overallDataCells = document.querySelectorAll(`td.${config.overallColClass}`); // S1/S2/S3 data cells
            const detailDataCells = document.querySelectorAll(`td.${config.detailColClass}`); // NR, NCL etc. data cells


            if (sec === sectionName) { // This is the section being toggled
                const isCurrentlyExpanded = overallHeaderInSecondRow.classList.contains('hidden'); // Check if overall is hidden (meaning details are visible)

                if (isCurrentlyExpanded) { // If currently expanded, collapse it
                    // Show overall header and hide details headers
                    overallHeaderInSecondRow.classList.remove('hidden');
                    detailHeaders.forEach(h => h.classList.add('hidden'));

                    // Show overall data cells and hide details data cells
                    overallDataCells.forEach(cell => cell.classList.remove('hidden'));
                    detailDataCells.forEach(cell => cell.classList.add('hidden'));

                    // Adjust top header colspan
                    topHeaderCell.setAttribute('colspan', '1');
                } else { // If currently collapsed, expand it
                    // Hide overall header and show details headers
                    overallHeaderInSecondRow.classList.add('hidden');
                    detailHeaders.forEach(h => h.classList.remove('hidden'));

                    // Hide overall data cells and show details data cells
                    overallDataCells.forEach(cell => cell.classList.add('hidden'));
                    detailDataCells.forEach(cell => cell.classList.remove('hidden'));

                    // Adjust top header colspan
                    topHeaderCell.setAttribute('colspan', config.detailColspan);
                }
            } else { // Other sections should be collapsed
                // Show overall header and hide details headers
                overallHeaderInSecondRow.classList.remove('hidden');
                detailHeaders.forEach(h => h.classList.add('hidden'));

                // Show overall data cells and hide details data cells
                overallDataCells.forEach(cell => cell.classList.remove('hidden'));
                detailDataCells.forEach(cell => cell.classList.add('hidden'));

                // Adjust top header colspan
                topHeaderCell.setAttribute('colspan', '1');
            }
        });
        setTimeout(adjustColumnWidths, 0); // Re-adjust column widths after toggling
    }


    function resetTable() {
        // Ensure all sections are collapsed and display their overall columns
        const sections = ['s1', 's2', 's3'];
        sections.forEach(sec => {
            const config = sectionConfig[sec];
            const topHeaderCell = document.querySelector(`.${config.topCellClass}`);
            const overallHeaderInSecondRow = document.querySelector(`th[data-key="${config.overallKey || sec.toUpperCase()}"]`);
            const detailHeaders = document.querySelectorAll(`th.${config.detailColClass}`);

            const overallDataCells = document.querySelectorAll(`td.${config.overallColClass}`);
            const detailDataCells = document.querySelectorAll(`td.${config.detailColClass}`);

            // Hide all detail headers and data cells
            detailHeaders.forEach(h => h.classList.add('hidden'));
            detailDataCells.forEach(cell => cell.classList.add('hidden'));

            // Show overall headers and data cells
            overallHeaderInSecondRow.classList.remove('hidden');
            overallDataCells.forEach(cell => cell.classList.remove('hidden'));

            // Reset top header colspan
            topHeaderCell.setAttribute('colspan', '1');
        });

        // Sort by "Overall" (Ov...) descending by default after reset
        const overallSortHeader = document.querySelector('th[data-key="Overall"].sortable');
        if (overallSortHeader) {
            sortTable(overallSortHeader, true); // true for initial descending sort
        }

        setTimeout(adjustColumnWidths, 0);
    }


    function sortTable(header, forceDescending = false) {
        const table = document.getElementById('prmbench-table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));

        const dataKey = header.dataset.key; // Get the data-key from the header
        const sortType = header.dataset.sort;

        let isDescending = forceDescending || (header === currentSortHeader && currentSortDirection === 'asc');

        rows.sort((a, b) => {
            // Find the corresponding original data rows using the Model name from the first cell
            const aModelName = a.querySelector('td:first-child a')?.textContent || a.querySelector('td:first-child')?.textContent;
            const bModelName = b.querySelector('td:first-child a')?.textContent || b.querySelector('td:first-child')?.textContent;

            const aData = window.prmLeaderboardData.find(item => item.Model === aModelName);
            const bData = window.prmLeaderboardData.find(item => item.Model === bModelName);

            if (!aData || !bData) return 0;

            let aValue = aData[dataKey];
            let bValue = bData[dataKey];

            if (sortType === 'number') {
                aValue = parseFloat(aValue) || -Infinity;
                bValue = parseFloat(bValue) || -Infinity;
                return isDescending ? (bValue - aValue) : (aValue - bValue);
            } else { // String sorting
                return isDescending
                    ? bValue.localeCompare(aValue)
                    : aValue.localeCompare(bValue);
            }
        });

        // Remove existing sort indicators from all headers in the second row
        document.querySelectorAll('#prmbench-table thead tr:last-child th').forEach(th => th.classList.remove('asc', 'desc'));

        // Add new sort indicator
        header.classList.add(isDescending ? 'desc' : 'asc');

        // Update current sort state
        currentSortHeader = header;
        currentSortDirection = isDescending ? 'desc' : 'asc';

        // Reappend sorted rows
        rows.forEach(row => tbody.appendChild(row));

        setTimeout(adjustColumnWidths, 0);
    }

    function initializeSorting() {
        // Sort by "Overall" (Ov...) descending on initial load
        const overallHeader = document.querySelector('th[data-key="Overall"].sortable');
        if (overallHeader) {
            sortTable(overallHeader, true);
        }
    }

    function adjustColumnWidths() {
        const table = document.getElementById('prmbench-table');
        if (!table) return;

        const headersRow2 = Array.from(table.querySelectorAll('thead tr:last-child th'));
        const bodyRows = Array.from(table.querySelectorAll('tbody tr'));

        const measureSpan = document.createElement('span');
        measureSpan.style.visibility = 'hidden';
        measureSpan.style.position = 'absolute';
        measureSpan.style.whiteSpace = 'nowrap';
        document.body.appendChild(measureSpan);

        headersRow2.forEach((header, colIndex) => {
            // Find the *corresponding* data cells in the tbody.
            // This is crucial because column indices can change with hidden cells.
            // We need to match based on the header's position in the second header row.

            if (!header.classList.contains('hidden')) {
                measureSpan.textContent = header.textContent;
                let maxWidth = measureSpan.offsetWidth;

                bodyRows.forEach(row => {
                    // Get the cell at the exact same visual index in the tbody row
                    const cell = row.cells[colIndex]; // row.cells automatically returns visible cells

                    if (cell && !cell.classList.contains('hidden')) {
                        measureSpan.innerHTML = cell.innerHTML; // Use innerHTML to account for rich text (bold, underline)
                        const cellWidth = measureSpan.offsetWidth;
                        if (cellWidth > maxWidth) {
                            maxWidth = cellWidth;
                        }
                    }
                });

                // Apply width to header and all its corresponding data cells
                const finalWidth = `${maxWidth + 16}px`; // Add padding

                header.style.width = header.style.minWidth = header.style.maxWidth = finalWidth;

                document.querySelectorAll(`#prmbench-table tbody tr td:nth-child(${colIndex + 1}):not(.hidden)`).forEach(cell => {
                    cell.style.width = cell.style.minWidth = cell.style.maxWidth = finalWidth;
                });
            }
        });

        document.body.removeChild(measureSpan);
    }


    function prepareScoresForStyling(data) {
        const scores = {};
        const fieldsToStyle = ['Overall', 'NR', 'NCL', 'S1', 'ES', 'SC', 'DC', 'CI', 'S2', 'PS', 'DR', 'MS', 'S3'];

        fieldsToStyle.forEach(field => {
            const values = data
                .map(row => parseFloat(row[field]))
                .filter(value => !isNaN(value));

            if (values.length > 0) {
                const sortedUniqueValues = [...new Set(values)].sort((a, b) => b - a);
                scores[field] = data.map(row => {
                    const value = parseFloat(row[field]);
                    if (isNaN(value)) {
                        return -1;
                    }
                    return sortedUniqueValues.indexOf(value);
                });
            } else {
                scores[field] = data.map(() => -1);
            }
        });
        return scores;
    }

    function applyStyle(value, rank) {
        if (value === undefined || value === null || value === '-') return '-';

        if (rank === 1) {
            // Best score: bold and red
            return `<b class="best-score-text red-bold-score">${value}</b>`;
        }
        if (rank === 2) {
            // Second best score: underline
            return `<u class="second-best-score-text">${value}</u>`;
        }
        return `${value}`;
    }

    // Initial call to fetch data and start the process
    fetchDataAndLoadTable();
});