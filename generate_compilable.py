# Module
import os

# Parameters
## Folder names
folder_names = ['chapters', 'notes']
## Compilable Flag
compilable_flag = '-compilable'


# Main
if __name__ == '__main__':
    # List files
    for folder_name in folder_names:
        # List files
        file_names = os.listdir( folder_name )
        # Select compilables
        file_paths = [ os.path.join(folder_name, file_name) for file_name in file_names if( compilable_flag in file_name ) and not( '.swp' in file_name ) ]
        # Read, Filter, Create
        for file_path in file_paths:
            # Report
            print( file_path + ' '*30 )
            # Read
            with open(file_path,'r') as in_file:
                file_content = in_file.read()
            # Filter
            file_content = file_content.split( '\\begin{document}' , 1)[1]
            file_content = file_content.split( '\\end{document}' , 1)[0]
            # OSX Compatibility
            osx_header = '% !TEX encoding = UTF-8 Unicode\n'
            file_content = osx_header + file_content
            # Create
            new_file_path = file_path.replace(compilable_flag,'')
            with open(new_file_path,'w') as out_file:
                out_file.write(file_content)
    # Last report
    # if( 'file_path' in locals() ):
    #     print( file_path + ' '*30 )
