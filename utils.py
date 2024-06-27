def plot_images_with_boxes_and_masks(images, targets, num_images=4):
    fig, axs = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
    if num_images == 1:
        axs = [axs]  # Ensure axs is iterable when num_images=1

    for idx in range(min(num_images, len(images))):
        img = (
            images[idx].permute(1, 2, 0).clone().detach().cpu().numpy()
        )  # Convert image from (C, H, W) to (H, W, C)
        axs[idx].imshow(img, cmap="gray")  # Display the background image

        if "masks" in targets[idx]:
            all_masks = targets[idx]["masks"]
            for mask in all_masks:
                mask = (
                    mask.squeeze()
                )  # Assuming masks are (1, H, W) and removing singular dimensions
                rgba_mask = np.zeros((*mask.shape, 4))  # Create an RGBA mask

                rgba_mask[:, :, 2] = 1.0  # Blue channel
                mask_data = mask.clone().detach().cpu().numpy()
                rgba_mask[:, :, 3] = (
                    mask_data * 0.1
                )  # Alpha channel, scale mask by 0.5 for transparency

                # Overlay the color mask with transparency where mask values are zero
                axs[idx].imshow(rgba_mask)

        # Draw bounding boxes
        for box in targets[idx]["boxes"]:
            box = box.cpu().numpy()
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor="r",
                facecolor="none",
            )
            axs[idx].add_patch(rect)

        axs[idx].axis("off")
    plt.tight_layout()
    return fig