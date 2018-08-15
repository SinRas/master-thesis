# Module
import os

# Where to Look
## Folders
search_folders = [ 'chapters', 'notes' ]
## Files
root_files = [ 'main.tex', 'commands.cmd', 'styles.sty' ]

# Add Header
def modify_file( file_path ):
    # Header
    osx_header = '% !TEX encoding = UTF-8 Unicode\n'
    # Read File
    with open(file_path,'r') as in_file:
        file_content = in_file.read()
    # Modify
    if( not( osx_header in file_content ) ):
        file_content = osx_header + file_content
    # Store File
    with open(file_path,'w') as out_file:
        out_file.write(file_content)
    # Return
    return

# Main
if __name__ == '__main__':
    # Base Files
    folder_name = '.'
    # Folder Name
    print('--> {}'.format( folder_name ))
    # Loop over Files
    for file_name in root_files:
        # File Path
        file_path = os.path.join( folder_name, file_name )
        # Report
        print( file_path )
        # Add Header
        modify_file( file_path )
    # Search Folders
    for folder_name in search_folders:
        # Folder Name
        print('--> {}'.format( folder_name ))
        # List files
        file_names = [ file_name for file_name in os.listdir(folder_name) if( file_name[-4:] == '.tex' ) ]
        # Loop over Files
        for file_name in file_names:
            # File Path
            file_path = os.path.join( folder_name, file_name )
            # Report
            print( file_path )
            # Add Header
            modify_file( file_path )