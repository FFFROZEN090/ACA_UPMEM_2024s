user=$(whoami)
image_name=aca
image_id=$(docker images | grep $image_name | awk '{print $3}')

dir=/home/$user/ACA_UPMEM_2024s/
workdir=/home/$user/ACA_UPMEM_2024s/


# Check if the image_id is not empty
if [ -z "$image_id" ]; then
  echo "Error: Image $image_name not found."
  exit 1
fi

docker run -it --rm -v ${dir}:${workdir} -w ${workdir} ${image_id}


