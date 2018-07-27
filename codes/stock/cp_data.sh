#!/bin/bash
# Parameters
SOURCE_DIR='/media/sinras/DATA/DataPerSymbol'
DESTINATION_DIR='data/iran'
FILE_NAME='daily_historical.csv.bz2'
# Copy Files
for FOLDER_PATH in $SOURCE_DIR/* ; do
	if [ -d "$FOLDER_PATH" ] && [ -f "$FOLDER_PATH/$FILE_NAME" ]; then
		cp "$FOLDER_PATH/$FILE_NAME" "$DESTINATION_DIR/$(basename $FOLDER_PATH).csv.bz2"
	fi
done
