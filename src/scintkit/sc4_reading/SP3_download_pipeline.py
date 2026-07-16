from SP3_download_pipeline import download_sp3_files

start_date = input("Start date (YYYY-MM-DD): ").strip()
end_date = input("End date (YYYY-MM-DD): ").strip()

download_sp3_files(
    start_date=start_date,
    end_date=end_date,
    output_dir=binary_dir,
)