import { Composition } from "remotion";
import { ArchitectureFlow } from "./ArchitectureFlow";

export const RemotionRoot = () => {
  return (
    <>
      <Composition
        id="ArchitectureFlow"
        component={ArchitectureFlow}
        durationInFrames={900} // 30s at 30fps
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
