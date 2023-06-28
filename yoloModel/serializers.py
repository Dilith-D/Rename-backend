from rest_framework import serializers

class ImageSerializer(serializers.Serializer):
    image = serializers.ImageField()

    def validate_image(self, value):
        valid_extensions = ['jpg', 'jpeg', 'png']
        max_size = 10 * 1024 * 1024  # 10 MB

        # Check if the file extension is valid
        file_extension = value.name.split('.')[-1].lower()
        if file_extension not in valid_extensions:
            raise serializers.ValidationError('Invalid file extension. Only JPEG, JPG, and PNG files are allowed.')

        # Check if the file size is within the limit
        if value.size > max_size:
            raise serializers.ValidationError('File size exceeds the limit. Maximum allowed size is 10 MB.')

        return value
