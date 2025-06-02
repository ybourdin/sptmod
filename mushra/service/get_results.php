<?php
header('Content-Type: text/csv');
header('Content-Disposition: attachment; filename="mushra.csv"');

if (!isset($_GET['testid'])) {
    http_response_code(400);
    exit("Error: Missing 'testid' parameter.");
}

$testid = preg_replace('/[^a-zA-Z0-9_]/', '', $_GET['testid']);
$pattern = "../results/" . $testid . "_*/mushra.csv";
$files = glob($pattern);

if (!$files) {
    exit("No matching files found.");
}

$output = ""; // Initialize output string
$headerPrinted = false;

foreach ($files as $file) {
    if (file_exists($file)) {
        $fileContent = file($file); // Read file into an array

        if (!$headerPrinted) {
            // Include the entire content of the first file
            $output .= implode("", $fileContent);
            $headerPrinted = true;
        } else {
            // Skip the first line (header) for subsequent files
            array_shift($fileContent); // Remove the first line
            $output .= implode("", $fileContent);
        }
    }
}

echo $output; // Output the concatenated content
exit();